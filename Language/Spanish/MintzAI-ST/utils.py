import json
import pandas as pd
import os
import logging
from pathlib import Path
from tqdm import tqdm
import codecs
import time

pd.set_option('display.max_rows', None)   # Display all rows
pd.set_option('display.max_columns', None) # Display all columns
pd.set_option('display.width', None)      # Ensure the display isn't truncated
pd.set_option('display.max_colwidth', None)

logger = logging.getLogger("audio_processing")

#################
# PREPROCESSING #
#################
def db_name(path):
    return Path(path).parts[3].split('.')[0]

def create_dir(path):
    database = db_name(path)
    if not os.path.exists(f'{database}/logs/'):
        os.makedirs(f'{database}/logs/')
    if not os.path.exists(f'{database}/csv_results/'):
        os.makedirs(f'{database}/csv_results/')
    return database

def load_data(stt, prompt_path, wave_path):
    wav_files = [file for file in os.listdir(wave_path) if file.endswith('.m4a')]
    if not wav_files:
        raise ValueError("No audio file found in the directory.")
    
    transcripts = []
    for wav_file in wav_files:
        txt_filename = wav_file.replace('.m4a', '.es')
        txt_filepath = os.path.join(prompt_path, txt_filename)
        
        if not os.path.exists(txt_filepath):
            raise ValueError(f"Transcription file {txt_filepath} does not exist.")
        
        with open(txt_filepath, 'r', encoding='utf-8') as txt_file:
            transcript = txt_file.readline().strip()
            transcripts.append(transcript)

    if len(wav_files) != len(transcripts):
        raise ValueError("The number of lines in the txt file doesn't match the number of wav files.")
        
    validation_df = pd.DataFrame({
        'wav_filename': wav_files,
        'transcript': transcripts
    })

    total_words = validation_df['transcript'].apply(lambda x: len(stt.transformation(x).split())).sum()
    return validation_df, total_words

####################
# PROCESSING AUDIO #
####################
def transcribe_audio(stt, audio_path, reference):
    try:
        hypothesis = stt.transcribe(audio_path)
    except FileNotFoundError:
        logger.info(f"File {audio_path} does not exist. Skipping.")
        return None

    reference_transformed = stt.transformation(reference)
    hypothesis_transformed = stt.transformation(hypothesis)
    
    if not reference_transformed:
        logger.warning(f"Empty reference for audio: {audio_path}. Skipping WER calculation.")
        return None

    wer = stt.compute_wer(reference_transformed, hypothesis_transformed)
    word_count = stt.compute_word_count(reference_transformed)
    error_count = stt.compute_error_count(wer, word_count)
    
    return wer, word_count, reference_transformed, hypothesis_transformed, int(error_count)

def process_audios(stt, validation_df, total_audios, path):
    results_df = pd.DataFrame(columns=['audio_file', 'reference', 'hypothesis', 'wer', 'words', 'errors'])
    n_audios = 10
    # for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
    for idx, row in tqdm(validation_df.head(n_audios).iterrows(), total=n_audios, desc="Processing audios"):
        audio_file = row['wav_filename']
        reference = row['transcript']
        audio_path = path / audio_file

        result = transcribe_audio(stt, audio_path, reference)
        if result is not None:
            wer, word_count, reference_transformed, hypothesis_transformed, error_count = result
            results_df.loc[idx] = [audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count]
            processing_info(idx+1, total_audios, audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count)
    
    return results_df

def calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database):
    total_errors = results_df['errors'].sum()
    wwer = total_errors / total_words
    mean_wer = results_df['wer'].mean()

    wwer_info(total_audios, total_words, total_errors, wwer, mean_wer)
    save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database)


##################
# SAVING RESULTS #
##################
def save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database):
    final_results_df = pd.DataFrame({
        'model': [stt.config['name']],
        'version': [stt.config['version']],
        'database': [database],
        'total_audios': [total_audios],
        'total_words': [total_words],
        'total_errors': [total_errors],
        'wwer': [wwer],
        'wer': [mean_wer]
    })
    
    file_name = '{}/csv_results/{}_{}_{}.csv'.format(database, stt.config['name'], stt.config['version'], database)
    with open(file_name, 'w') as file:
        final_results_df.to_csv(file, index=False)


######################
# LOGGER INFORMATION #
######################
def wwer_info(total_audios, total_words, total_errors, wwer, mean_wer):
    logger.info("\nWWER INFO")
    logger.info(f'Total audios: \t{total_audios}')
    logger.info(f'Total words: \t{total_words}')
    logger.info(f"Total errors: \t{total_errors}")
    logger.info(f"Weighted WER: \t{wwer}")
    logger.info(f"WER: \t{mean_wer}")

def header_info(stt, path, total_audios, total_words):
    logger.info("\nHEADER INFO")
    logger.info(f"Model:\t\t {stt.config['name']}")
    logger.info(f"Version:\t {stt.config['version']}")
    logger.info(f"Main DB:\t {db_name(path)}")
    logger.info(f'Total audios:\t {total_audios}')
    logger.info(f"Total words:\t {total_words}")

def processing_info(idx, total_audios, audio_file, reference, hypothesis, wer, word_count, error_count):
    logger.info("\nPROCESSING INFO")
    logger.info(f"Processing audio #{idx} of {total_audios}")
    logger.info(f"Audio file: {audio_file}")
    logger.info(f"\tReference: \t\t\t{reference}")
    logger.info(f"\tHypothesis: \t\t{hypothesis}")
    logger.info(f"\tWord Error Rate: \t\t\t{wer}")
    logger.info(f"\tWords in reference: \t\t{word_count}")
    logger.info(f"\tWrong Words: \t\t\t\t{error_count}")
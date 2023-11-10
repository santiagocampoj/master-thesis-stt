import time
import pandas as pd
import os
import logging
from pathlib import Path
from tqdm import tqdm
import codecs
pd.set_option('display.max_rows', None)   # Display all rows
pd.set_option('display.max_columns', None) # Display all columns
pd.set_option('display.width', None)      # Ensure the display isn't truncated
pd.set_option('display.max_colwidth', None)

logger = logging.getLogger("audio_processing")

#################
# PREPROCESSING #
#################
def db_name(path):
    return Path(path).parts[6].split('.')[0]

def create_dir(path):
    database = db_name(path)
    if not os.path.exists(f'{database}/logs/'):
        os.makedirs(f'{database}/logs/')
    if not os.path.exists(f'{database}/csv_results/'):
        os.makedirs(f'{database}/csv_results/')
    return database

def load_data(prompt_path, wave_path):
    all_data = []
    for txt_file in os.listdir(prompt_path):
        print(f"Processing ---> {txt_file}")
        with codecs.open(os.path.join(prompt_path, txt_file), 'r', encoding='utf-8-sig') as file:
            raw_lines = file.readlines()
            for line in raw_lines:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[1].startswith("G"):
                    start_time = float(parts[0].replace('[', '').split(',')[0])
                    end_time = float(parts[0].replace(']', '').split(',')[1])
                    transcription = " ".join(parts[3:])
                    full_audio_path = os.path.join(wave_path, txt_file.replace('.txt', '.wav'))
                    if os.path.exists(full_audio_path):
                        all_data.append({
                            'wav_filename': full_audio_path,
                            'transcript': transcription,
                            'start_time': start_time,
                            'end_time': end_time
                        })
    df = pd.DataFrame(all_data)
    total_words = df['transcript'].apply(lambda x: len(x.split())).sum()
    return df, int(total_words)

####################
# PROCESSING AUDIO #
####################
def transcribe_audio(stt, audio_path, reference, start_time, end_time):
    try:
        hypothesis = stt.transcribe(audio_path, start_time=start_time, end_time=end_time)
    except FileNotFoundError:
        logger.warning(f"File {audio_path} does not exist. Skipping.")
        return None

    reference_transformed = stt.transformation(reference)
    hypothesis_transformed = stt.transformation(hypothesis)

    if not reference_transformed:
        logger.warning(f"Empty reference for audio: {audio_path}. Skipping WER calculation.")
        return None

    wer = stt.compute_wer(reference_transformed, hypothesis_transformed)
    
    word_count = stt.compute_word_count(reference_transformed)
    error_count = stt.compute_error_count(wer, word_count)
    
    return wer, word_count, reference_transformed, hypothesis_transformed, error_count

def process_audios(stt, validation_df, total_audios, path):
    print("\nEntering Process Audios")
    results_df = pd.DataFrame(columns=['audio_file', 'reference', 'hypothesis', 'wer', 'words', 'errors'])

    for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
    # for idx, row in tqdm(validation_df.head(5).iterrows(), total=total_audios, desc="Processing audios"):
        audio_file = row['wav_filename']
        reference = row['transcript']
        audio_path = path / audio_file

        result = transcribe_audio(stt, audio_path, reference, row['start_time'], row['end_time'])
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
        'mean_wer': [mean_wer]
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
    logger.info(f"Mean WER: \t{mean_wer}")

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
    logger.info(f"\tWrong Words: \t\t\t{error_count}")
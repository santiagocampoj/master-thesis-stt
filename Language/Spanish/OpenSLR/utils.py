import glob
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

#################
# PREPROCESSING #
#################
def db_name(path):
    return Path(path).parts[3]

def create_dir(path):
    database = db_name(path)
    if not os.path.exists(f'{database}/logs/'):
        os.makedirs(f'{database}/logs/')
    if not os.path.exists(f'{database}/results/'):
        os.makedirs(f'{database}/results/')
    return database

def flac_txt_files(directory):
    flac_files = glob.glob(os.path.join(directory, '*.flac'))
    txt_files = [f"{os.path.splitext(flac_file)[0]}.txt" for flac_file in flac_files]
    total_audios = len(flac_files)
    return list(zip(flac_files, txt_files)), total_audios

####################
# PROCESSING AUDIO #
####################
def transcribe_audio(stt, audio_path, reference, logger):
    try:
        hypothesis = stt.run(audio_path)
    except FileNotFoundError:
        logger.info(f"File {audio_path} does not exist. Skipping.")
        return None
    except OSError as e:
        logger.error(f"OS error occurred when processing file {audio_path}: {e}")
        return None

    reference_transformed = stt.transformation(reference)
    hypothesis_transformed = stt.transformation(hypothesis)

    wer = stt.compute_wer(reference_transformed, hypothesis_transformed)
    word_count = stt.compute_word_count(reference_transformed)
    error_count = stt.compute_error_count(wer, word_count)
    
    return wer, word_count, reference_transformed, hypothesis_transformed, error_count

def process_audios(stt, directory, total_words, logger):
    file_pairs, total_audios = flac_txt_files(directory)
    results_df = pd.DataFrame(columns=['audio_file', 'reference', 'hypothesis', 'wer', 'words', 'errors'])

    for idx, (audio_file, txt_file) in tqdm(enumerate(file_pairs), total=total_audios, desc="Processing audios"):
        if not os.path.exists(txt_file):
            logger.warning(f"Text file {txt_file} does not exist. Skipping audio {audio_file}.")
            continue
        
        with open(txt_file, 'r') as f:
            reference = f.read().strip()

        result = transcribe_audio(stt, audio_file, reference, logger)
        if result is not None:
            wer, word_count, reference_transformed, hypothesis_transformed, error_count = result
            results_df.loc[idx] = [audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count]
            processing_info(idx+1, total_audios, audio_file, reference_transformed, hypothesis_transformed, wer, word_count, error_count, logger)
    
    return results_df

def calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, logger):
    total_errors = results_df['errors'].sum()
    wwer = total_errors / total_words
    mean_wer = results_df['wer'].mean()

    wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger)
    save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, logger)


##################
# SAVING RESULTS #
##################
def save_final_results(stt, total_audios, total_words, total_errors, wwer, mean_wer, audio_path, database, logger):
    final_results_df = pd.DataFrame({
        'model': [stt.config['name'].replace(" ", "_")],
        'language': [stt.lang],
        'database': [database],
        'total_audios': [total_audios],
        'total_words': [total_words],
        'total_errors': [total_errors],
        'wwer': [wwer],
        'mean_wer': [mean_wer]
    })

    file_name = f"{database}/results/{stt.config['name'].replace(' ', '_')}_{database}.csv"
    with open(file_name, 'w') as file:
        final_results_df.to_csv(file, index=False)


######################
# LOGGER INFORMATION #
######################
def wwer_info(total_audios, total_words, total_errors, wwer, mean_wer, logger):
    logger.info("\nWWER INFO")
    logger.info(f'Total audios: \t{total_audios}')
    logger.info(f'Total words: \t{total_words}')
    logger.info(f"Total errors: \t{total_errors}")
    logger.info(f"Weighted WER: \t{wwer}")
    logger.info(f"WER: \t{mean_wer}")

def header_info(stt, path, total_audios, total_words, logger):
    logger.info("\nHEADER INFO")
    logger.info(f"Model:\t\t {stt.config['name']}")
    logger.info(f"Version:\t {stt.config['version']}")
    logger.info(f"Main DB:\t {db_name(path)}")
    logger.info(f'Total audios:\t {total_audios}')
    logger.info(f"Total words:\t {total_words}")

def processing_info(idx, total_audios, audio_file, reference, hypothesis, wer, word_count, error_count, logger):
    logger.info("\nPROCESSING INFO")
    logger.info(f"Processing audio #{idx} of {total_audios}")
    logger.info(f"Audio file: {audio_file}")
    logger.info(f"\tReference: \t\t\t{reference}")
    logger.info(f"\tHypothesis: \t\t{hypothesis}")
    logger.info(f"\tWord Error Rate: \t\t\t{wer}")
    logger.info(f"\tWords in reference: \t\t{word_count}")
    logger.info(f"\tWrong Words in hypothesis: \t{error_count}")
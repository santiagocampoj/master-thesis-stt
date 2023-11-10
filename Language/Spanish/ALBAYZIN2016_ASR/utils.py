import pandas as pd
import os
import logging
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re

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

def parse_xml(xml_file):
    """Parse the XML file and return the transcription data."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_data = []

    for unit in root.findall(".//UNIT"):
        # print(unit)
        start_time = float(unit.attrib["startTime"])
        # print(start_time)
        end_time = float(unit.attrib["endTime"])
        # print(end_time)
        transcription = unit.text if unit.text else ""  # noneyype
        transcription = transcription.strip()
        # print(transcription)
        all_data.append({
            'transcript': transcription,
            'start_time': start_time,
            'end_time': end_time
        })

    return all_data

def clean_transcriptions(transcription):
    # Special character removal
    transcription = transcription.replace("¤", "").replace("=", "").replace("xxx", "").replace("hhh", "")
    
    # alt notations removal
    alt_pattern = re.compile(r'{%.*?%}')
    transcription = re.sub(alt_pattern, '', transcription)

    # Comments removal
    comment_pattern = re.compile(r'\{.*?\}')
    transcription = re.sub(comment_pattern, '', transcription)

    # Single and double slashes removal
    transcription = transcription.replace(" / ", " ").replace(" // ", " ").replace("[/]", "")

    # Other replacements
    transcription = transcription.replace(">", "").replace("eh", "").replace("mm", "").replace("&eh", "").replace("ah e a", "")

    # Strip extra whitespaces
    transcription = " ".join(transcription.split()).strip()
    transcription = transcription.lstrip()

    return transcription

def load_data(stt, prompt_path, wave_path):
    all_data = []
    
    for xml_file in os.listdir(prompt_path):
        if xml_file.endswith('.xml'):
            print(f"Processing ---> {xml_file}")
            parsed_data = parse_xml(os.path.join(prompt_path, xml_file))
            for entry in parsed_data:
                entry['transcript'] = clean_transcriptions(entry['transcript'])  # Cleaning text
                full_audio_path = os.path.join(wave_path, xml_file.replace('.xml', '.wav'))
                if os.path.exists(full_audio_path):
                    entry['wav_filename'] = full_audio_path
                    all_data.append(entry)

    df = pd.DataFrame(all_data)
    total_words = df['transcript'].apply(lambda x: len(stt.transformation(x).split())).sum()
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
    results_df = pd.DataFrame(columns=['audio_file', 'reference', 'hypothesis', 'wer', 'words', 'errors'])

    # for idx, row in tqdm(validation_df.iterrows(), total=total_audios, desc="Processing audios"):
    for idx, row in tqdm(validation_df.head(50).iterrows(), total=total_audios, desc="Processing audios"):
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
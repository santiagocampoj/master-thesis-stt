from model_config_general import SPANISH_STT
from stt_class_general import STT

from .utils import *

from pathlib import Path
import logging
import argparse

logger = logging.getLogger("pydub.converter")
logger.setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-d', '--db-directory', required=True, help='Path to database files directory.')
    args = parser.parse_args()

    stt = STT(SPANISH_STT)
    stt.load_model()

    path = Path(args.db_directory)
    database, sub_database, section = create_dir(path)
    
    logger = logging.getLogger("audio_processing")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{database}/logs/{sub_database}/{section}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    validation_df, total_words = load_data(stt, path)
    
    total_audios = len(validation_df)
    header_info(stt, path, total_audios, total_words)
    
    results_df = process_audios(stt, validation_df, total_audios, path)
    calculate_wwer(stt, results_df, total_audios, total_words, path, database, sub_database, section)

if __name__ == "__main__":
    main()
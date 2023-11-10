from model_config_general import SPANISH_STT
from stt_class_general import STT

from .utils import *

from pathlib import Path
import logging
import argparse

logger = logging.getLogger("pydub.converter")
logger.setLevel(logging.WARNING)

def main():
    # set parser
    parser = argparse.ArgumentParser(description="Insert the audio and text file to be processed.")
    parser.add_argument('-a', '--audio-path', required=True, help='Path to audio files directory or combined directory.')
    parser.add_argument('-t', '--text-path', help='Path to text metadata file. If not provided, assumes combined directory with --audio-path.')
    args = parser.parse_args()

    stt = STT(SPANISH_STT)
    stt.load_model()

    # Set the audio path
    audio_path = Path(args.audio_path)

    # create dir to save results and logs
    database, speaker = create_dir(audio_path)

    #set up the logger
    logger = logging.getLogger("audio_processing")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{database}/logs/{speaker}/{database}_{speaker}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Get general information based on whether --text-path is provided or not
    if args.text_path:
        validation_df, total_words = load_data(stt, args.text_path, args.audio_path)
    else:
        validation_df, total_words = load_data(stt, combined_path=args.audio_path)

    total_audios = len(validation_df)
    header_info(stt, audio_path, total_audios, total_words)
    
    # Processing audio files
    results_df = process_audios(stt, validation_df, total_audios, audio_path)

    # Saving results
    calculate_wwer(stt, results_df, total_audios, total_words, audio_path, database, speaker)

if __name__ == "__main__":
    main()

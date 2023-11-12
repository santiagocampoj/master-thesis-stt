import os
import time
import logging
import pydub
import numpy as np
import argparse
from stt import Model
from coqui_stt_model_manager.modelmanager import ModelManager
from pydub import AudioSegment
from pydub.silence import split_on_silence

STT_HOST = 'https://coqui.gateway.scarf.sh'
STT_HOST_AHOLAB = 'https://aholab.ehu.eus/~xzuazo/models'
STT_MODELS = {

    'eu': {
        'name': 'Basque STT v0.1.7',
        'language': 'Basque',
        'version': 'v0.1.7',
        'creator': 'ITML',
        'acoustic': f'{STT_HOST_AHOLAB}/Basque STT v0.1.7/model.tflite',
        'scorer': f'{STT_HOST_AHOLAB}/Basque STT v0.1.7/kenlm.scorer',
    },
    'es': {
        'name': 'Spanish STT v0.0.1',
        'language': 'Spanish',
        'version': 'v0.0.1',
        'creator': 'Jaco-Assistant',
        'acoustic': f'{STT_HOST}/spanish/jaco-assistant/v0.0.1/model.tflite',
        'scorer': f'{STT_HOST}/spanish/jaco-assistant/v0.0.1/kenlm_es.scorer',
    },
}

INSTALL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'stt'
)

def ensure_samplerate(audio_path, desired_sample_rate):
    try:
        sound = pydub.AudioSegment.from_file(audio_path)
    except pydub.exceptions.CouldntDecodeError as error:
        raise ValueError('Could not decode audio file.') from error

    # get the frame rate and
    sample_rate = sound.frame_rate
    # check if it is the desired sample rate
    if sample_rate != desired_sample_rate:
        sound = sound.set_frame_rate(desired_sample_rate)
	# if so, save it in wav format (whatever)
        # sound.export(audio_path, format='wav')
    return sound

def ensure_mono(sound):
    if sound.channels > 1:  # force mono | channel one
        sounds = sound.split_to_mono() # split the audio into mono
        sound = sounds[0] # keep just the first channel
    return sound

def read_wav(audio_path, desired_sample_rate):
    try:
        sound = pydub.AudioSegment.from_file(audio_path)
    except pydub.exceptions.CouldntDecodeError as error:
        raise ValueError('Could not decode audio file.') from error
    
    sample_rate = sound.frame_rate
    if sample_rate != desired_sample_rate:
        sound = sound.set_frame_rate(desired_sample_rate)

    sound = ensure_samplerate(audio_path, desired_sample_rate)
    sound = ensure_mono(sound)
    audio = np.array(sound.get_array_of_samples())

    logging.debug(f"Audio type: {audio.dtype}")
    # It needs to be in 16-bit precision:
    if audio.dtype == 'int8':
        audio = np.array(audio.astype(float) * (2**8), dtype=np.int16)
    
    elif audio.dtype == 'int32':
        audio = np.array(audio.astype(float) / (2**16), dtype=np.int16)
    
    elif audio.dtype == 'float32':
        audio = np.array(audio.astype(float) / (2**16), dtype=np.int16)
    
    return audio

# def segment_audio(audio_path, min_silence_len=500, silence_thresh=-50, keep_silence=400):
def segment_audio(audio_path, min_silence_len=400, silence_thresh=-50, keep_silence=350):
    """
    Splits the audio file into chunks based on silence.
    
    :param audio_path: The path to the audio file.
    :param min_silence_len: (int) minimum length of a silence to be used for a split. Default to 1000ms.
    :param silence_thresh: (int) the upper bound for what is considered silence. Default to -40 dBFS.
    :param keep_silence: (int) amount of silence to leave at the beginning and end of each chunk. Default to 200ms.
    :return: List of AudioSegment instances representing the chunks.
    """
    sound = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    return chunks

def transcribe_chunks(stt, chunks, sample_rate):
    transcriptions = []
    for i, chunk in enumerate(chunks):
        chunk.export(f"/tmp/chunk_{i}.wav", format="wav")
        
        audio_data = read_wav(f"/tmp/chunk_{i}.wav", sample_rate)
        transcription = stt.model.stt(audio_data)
        transcriptions.append(transcription)
        
        os.remove(f"/tmp/chunk_{i}.wav")
    return transcriptions



class STT:
    def __init__(self, lang, scorer=True):
        self.lang = lang
        if self.lang not in STT_MODELS:
            raise ValueError(f'Unknown language: {self.lang}')
        self.config = STT_MODELS[self.lang]
        if not scorer and 'scorer' in self.config:
            del self.config['scorer']
        
        os.makedirs(INSTALL_DIR, exist_ok=True)
        self.model = None
        logging.debug('Downloading %s model...', lang)
        self.download()
        logging.debug('Model downloaded.')
        logging.debug('Loading %s model...', lang)
        self.load()
        logging.debug('Model loaded.')

    def download(self):
        self.manager = ModelManager(install_dir=INSTALL_DIR)
        self.manager.download_model(STT_MODELS[self.lang])
        if not self.config['name'] in self.manager.models_dict():
            logging.debug('Waiting for %s to download...', self.config['name'])
            while not self.config['name'] in self.manager.models_dict():
                time.sleep(1)
        self.card = self.manager.models_dict()[self.config['name']]

    def scorer(self, scorer_path):
        if scorer_path is None:
            self.model.disableExternalScorer()
        else:
            self.model.enableExternalScorer(scorer_path)

    def load(self):
        if self.model is not None:
            return
        acoustic_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.card.acoustic_path
        )
        self.model = Model(acoustic_path)
        if 'scorer' in self.config:
            scorer_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.card.scorer_path
            )
            self.scorer(scorer_path)

    def run(self, audio_path):
        logging.debug('[STT:%s] Audio path: %s', self.lang, audio_path)
        desired_sample_rate = self.model.sampleRate()
        audio = read_wav(audio_path, desired_sample_rate)
        text = self.model.stt(audio)
        logging.debug('[STT:%s] Transcription: "%s"', self.lang, text)
        return text

def main():
    parser = argparse.ArgumentParser(description='Transcribe an audio file.')
    parser.add_argument('audio_file', help='Path to the audio file to transcribe.')
    parser.add_argument('--language', '-l', default='en', choices=STT_MODELS.keys(), help='Language of the transcription model.')
    args = parser.parse_args()

    audio_path = args.audio_file
    language = args.language

    logging.basicConfig(level=logging.debug)
    logging.getLogger('pydub.converter').setLevel(logging.WARNING)

    stt = STT(language)

    try:
        threshold = 11
        sound = AudioSegment.from_file(audio_path)
        duration_seconds = len(sound) / 1000

        if duration_seconds > threshold:  # if the audio is longer than x seconds
            logging.debug(f"Segmenting file {audio_path} [ {duration_seconds}s ] because its duration is longer than {threshold} seconds...")
            chunks = segment_audio(audio_path)
            logging.debug(f"There are {len(chunks)} chunks taken from {audio_path.split('/')[-1]}")
            transcriptions = transcribe_chunks(stt, chunks, stt.model.sampleRate())
            full_transcript = ' '.join(transcriptions)
        else:
            logging.debug(f"Transcribing file {audio_path} as a single chunk because its duration is less than {threshold} seconds...")
            full_transcript = stt.run(audio_path)

        logging.debug("Transcription completed.")
        print(f"\nFull Transcription: \n{full_transcript}")
    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")

if __name__ == '__main__':
    main()

# run:
# python script_name.py path_to_audio_file.wav --language es

import os
import time
import logging
import pydub
import numpy as np
from stt import Model
from scipy.io import wavfile
from coqui_stt_model_manager.modelmanager import ModelManager
import argparse
import jiwer

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
    }
}

INSTALL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'Language', 'models'
)

def ensure_samplerate(audio_path, desired_sample_rate):
    try:
        sound = pydub.AudioSegment.from_file(audio_path)
    except pydub.exceptions.CouldntDecodeError as error:
        raise ValueError('Could not decode audio file.') from error

    sample_rate = sound.frame_rate
    if sample_rate != desired_sample_rate:
        sound = sound.set_frame_rate(desired_sample_rate)
        sound.export(audio_path, format='wav')


def read_wav(audio_path, desired_sample_rate):
    try:
        sound = pydub.AudioSegment.from_file(audio_path)
    except pydub.exceptions.CouldntDecodeError as error:
        raise ValueError('Could not decode audio file.') from error
    
    sample_rate = sound.frame_rate
    if sample_rate != desired_sample_rate:
        sound = sound.set_frame_rate(desired_sample_rate)
    
    if sound.channels > 1:
        sounds = sound.split_to_mono() 
        sound = sounds[0] 
    audio = np.array(sound.get_array_of_samples())

    # It needs to be in 16-bit precision:
    if audio.dtype == 'int8':
        audio = np.array(audio.astype(float) * (2**8), dtype=np.int16)
    elif audio.dtype == 'int32':
        audio = np.array(audio.astype(float) / (2**16), dtype=np.int16)
    elif audio.dtype == 'float32':
        audio = np.array(audio.astype(float) / (2**16), dtype=np.int16)
    
    return audio

class STT:
    def __init__(self, lang, scorer=True):
        self.lang = lang
        if self.lang not in STT_MODELS:
            raise ValueError(f'Unknown language: {self.lang}')
        self.config = STT_MODELS[self.lang]
        if not scorer and 'scorer' in self.config:
            del self.config['scorer']
        
        self.transformation = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        ])
        
        os.makedirs(INSTALL_DIR, exist_ok=True)

        self.model = None
        logging.info('Downloading %s model...', lang)
        self.download()
        logging.info('Model downloaded.')
        logging.info('Loading %s model...', lang)
        self.load()
        logging.info('Model loaded.')

    def download(self):
        self.manager = ModelManager(install_dir=INSTALL_DIR)
        self.manager.download_model(STT_MODELS[self.lang])

        if not self.config['name'] in self.manager.models_dict():
            logging.info('Waiting for %s to download...', self.config['name'])

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
        ) # join initial path + acoustic model path
        self.model = Model(acoustic_path)
        if 'scorer' in self.config:
            scorer_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.card.scorer_path
            )
            self.scorer(scorer_path)

    def run(self, audio_path):
        logging.debug('[STT:%s] Audio path: %s', self.lang, audio_path) # indicate language and audio
        desired_sample_rate = self.model.sampleRate() # change the sr
        audio = read_wav(audio_path, desired_sample_rate) # open the audio at the desired sr
        text = self.model.stt(audio) # transcription from the model
        # logging.debug('[STT:%s] Transcription: "%s"', self.lang, text) # output
        return text

    def compute_wer(self, reference, hypothesis):
        reference_transformed = self.transformation(reference)
        hypothesis_transformed = self.transformation(hypothesis)
        print(f"\nReference:\n{reference_transformed}")
        print(f"\nHypothesis:\n{hypothesis_transformed}")
        return jiwer.wer(reference_transformed, hypothesis_transformed)

    def compute_word_count(self, reference):
        reference_transformed = self.transformation(reference)
        words = len(reference_transformed.split())
        return words

    def compute_error_count(self, wer, word_count):
        return int(round(wer * word_count))

def get_args():
    parser = argparse.ArgumentParser(description="Transcribe an audio file using STT.")
    parser.add_argument('-a', '--audio', required=True, help="Path to the audio file to be transcribed.")
    parser.add_argument('-l', '--language', required=True, choices=['es', 'eu'], default='es', help="Language for the STT model. Choices are 'es' for Spanish or 'eu' for Basque. Default is 'es'.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    logging.basicConfig(level=logging.DEBUG) 
    stt_instance = STT(args.language)

    # reference_transcription = "Hezkuntzak prestatu zituen probak pisa eta antzekoak eredu."
    
    # reference_transcription = "Habita en aguas poco profundas y rocosas"
    # reference_transcription = "epa begira ba zera  galdera bat daukat CD bueno CD ez bideo  baten {FIL}  ba begira nago  ia aurkitzen dudan ez dakit deskatalogatuta egon den Stray Catsena da oraintxe elkartu behar direz gainera hogei urte pasata   baina  hau aurreko denborena da  {FRA} ez dakit ondo  zelako izena daukan bez ez baina ba dakit Japonian {FIL}  ba grabatu  egin  zela edo  denadelakoa esa ndakoa ez dakit  bere izenik {FIL}  ezta  noiz izan zen hogei urte  izango direz  igual {FIL} argitaratu  {FRA} zena Europan badakit {FRA} deskatalogatuta dagoela eta ba ez dakit {FIL}  Japo nian edo Amerikan lortu daitekeen {FIL}  jakin gura dut ia zeuek {FIL} lortu dezakezuen {FIL} bideo hori {FRA}  ba ia lan egiten duzuen Alemaniarekin oi Alemaniarekin   esango dut {FIL}  Japoniarekin ala Amerikarekin {FRA} ba ba hori  ba eskuratzeko mesedez ipini {FRA} nigaz kontaktuan nere telefonoa  bederatzi lau lau lau lau lau bost da eta ba hori gura neuke jakitea ahal ba da {FRA} ba ipini   nigaz mesedez kontaktuan ba nahiko {FRA}  interesa daukadalako opari batentzako delako aspaldi nago hori bilatzen {FRA} eta nahiko interesatuta nago  ez dakit diru aldetik zenbat izango dan {FRA}  horri horri buruz ere hitz egin gura nuke {FIL} oso garestia ba da ba noski ez dut hartuko eta   baina  bueno {FIL}  nahiko diru ordaintzeko prest nago mesedez deitu ondo"
    reference_transcription = "Señor Presidente, en primer lugar, quisiera felicitar al señor Seeber por el trabajo realizado, porque en su informe se recogen muchas de las preocupaciones manifestadas en esta Cámara en torno a estos problemas cruciales para toda la Unión Europea. Son éstos: la escasez de agua y la sequía, asuntos que ya han dejado de ser solamente un problema de los países del sur de Europa. Me alegro de que este proyecto incorpore alguna de las reflexiones realizadas en la opinión de la que fui ponente en la Comisión de Agricultura, a favor de una actividad agrícola, al destacar, por ejemplo, el papel que desempeñan los agricultores en la gestión sostenible de los recursos disponibles. Incluye, asimismo, una referencia a la sequía y a la escasez de agua como factores que agravan los precios de las materias primas, aspecto que, en los tiempos que corren, también me parece muy importante señalar para que tengamos presente no solamente la dimensión medioambiental de este problema, sino también algunas de sus consecuencias económicas más relevantes. La Comisión de Medio Ambiente incluyó, por otra parte, la idea de crear un observatorio europeo de la sequía, acción que se menciona, asimismo, en la opinión de la Comisión de Agricultura y que espero que, algún día, se vea reflejada en hechos concretos. El texto que se somete mañana a votación no recoge, sin embargo, una propuesta realizada por la Comisión de Agricultura para que se estudie la puesta en marcha de un fondo de adaptación económico contra la sequía, que beneficiaría a todos los sectores económicos, incluido también el de la agricultura. Por mi parte, me gustaría dejar claro que seguiré defendiendo la constitución de este fondo, una idea que volveré a proponer al Parlamento cuando llegue en los próximos meses la comunicación que tiene previsto elaborar la Comisión sobre la adaptación al cambio climático. Me parece que, después de que se constituyera el Fondo de Solidaridad, que en su día se creó para paliar las pérdidas provocadas por las calamidades climáticas, es hora de que pensemos en un instrumento que actúe y destinado a financiar medidas de prevención para reducir el coste medioambiental y económico de esas catástrofes climáticas."
    transcription = stt_instance.run(args.audio)

    # WER
    wer = stt_instance.compute_wer(reference_transcription, transcription)
    print("\nWord Error Rate:\n", wer)

    # word count
    word_count = stt_instance.compute_word_count(reference_transcription)
    print("\nWord Count:\n", word_count)

    # error count
    error_count = stt_instance.compute_error_count(wer, word_count)
    print("\nError Count:\n", error_count)

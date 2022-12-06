import os
from tqdm.notebook import tqdm
import librosa
import soundfile as sf
from utils import get_params
import warnings

extraction_params = get_params("./record_graph_construction/extraction_params.json")
SOURCE_PATH = extraction_params["source_path"]
FOLDER_PATH = extraction_params["folder_path"]
IMG_FOLDER_PATH = extraction_params["img_folder_path"]

def vid_2_flac(vid_path, aud_path):
    # load the audio from the video, resample to 16kHz and write to an flac audiofile
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     data, samplerate = librosa.load(vid_path, sr=16000)
    #     sf.write(aud_path, data, 16000, format="flac", subtype='PCM_16')
    data, samplerate = librosa.load(vid_path, sr=16000)
    sf.write(aud_path, data, 16000, format="flac", subtype='PCM_16')

def extract_all_audios():
    err = []
    for vid_file in tqdm(os.listdir(SOURCE_PATH)):
        vid_path = f'{SOURCE_PATH}/{vid_file}'
        aud_path = f'{FOLDER_PATH}/audio/{".".join(vid_file.split(".")[:-1])}.flac'
        if aud_path.split("/")[-1] not in os.listdir(f"{FOLDER_PATH}/audio/"):
            print(aud_path.split("/")[-1])
            try:
                vid_2_flac(vid_path, aud_path)
            except:
                err.append((vid_path, aud_path))


if __name__ == "__main__":
    print(extract_all_audios())
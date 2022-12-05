import os
from tqdm.notebook import tqdm
import librosa
import soundfile as sf

BATCH_SIZE = 64
DOWNSAMPLE = 12
SOURCE_PATH = "./videos"
FOLDER_PATH = "./results_1020"
IMG_FOLDER_PATH = FOLDER_PATH + "/keyframes"

def vid_2_flac(vid_path, aud_path):
    # load the audio from the video, resample to 16kHz and write to an flac audiofile
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

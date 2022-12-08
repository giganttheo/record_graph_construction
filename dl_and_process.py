from tqdm import tqdm
import json
import os

from audio_extraction import vid_2_flac

#from process_video import process_video
from utils import get_params

extraction_params = get_params("./record_graph_construction/extraction_params.json")
SOURCE_PATH = extraction_params["source_path"]
FOLDER_PATH = extraction_params["folder_path"]
IMG_FOLDER_PATH = extraction_params["img_folder_path"]

def process_video(vid_file):
    err_audio = []
    vid_path = f'{SOURCE_PATH}/{vid_file}'
    aud_path = f'{FOLDER_PATH}/audio/{".".join(vid_file.split(".")[:-1])}.flac'
    if aud_path.split("/")[-1] not in os.listdir(f"{FOLDER_PATH}/audio/"):
        print(aud_path.split("/")[-1])
        try:
            vid_2_flac(vid_path, aud_path)
        except:
            err_audio.append((vid_path, aud_path))
    return err_audio

if __name__ == "__main__":
    for vid_file in sorted(os.listdir(SOURCE_PATH)):
        process_video(vid_file)
    # with open("./record_graph_construction/dl_status.json", "r") as f:
    #     dl_status = json.load(f)
    # with open("./record_graph_construction/dl_status_local.json", "r") as f:
    #     dl_status_local = json.load(f)
    # for k in dl_status.keys():
    #     print(k)
    #     if not dl_status[k] and not dl_status_local[k]:
    #         os.system(f"wget -O ./videos/ {k}")
    #         dl_status_local[k] = True
    #         with open("./dl_status_local.json", "w") as f:
    #             f.dump(dl_status_local, f)
    #         err_audio, err_slides, err_transcription = process_video(k.split("/")[-1], None, False)
    #         if (len(*err_audio, *err_slides, *err_transcription)) > 0:
    #             print(err_audio, err_slides, err_transcription)
    #         dl_status_local[k] = True
    #         with open("./record_graph_construction/dl_status_local.json", "w") as f:
    #             json.dump(dl_status_local, f)
    #         os.system(f"rm ./videos/{k.split('/')[-1]}")

import random
import json
import os
import whisper

from feature_extraction.audio_extraction import vid_2_flac
from slide_extraction import compute_batch_hashes, compute_threshold, get_slides
from transcript_extraction import transcribe

#from process_video import process_video
from utils import get_params

extraction_params = get_params("./record_graph_construction/extraction_params.json")
SOURCE_PATH = extraction_params["source_path"]
FOLDER_PATH = extraction_params["folder_path"]
IMG_FOLDER_PATH = extraction_params["img_folder_path"]

def process_video(transcription_model, vid_file):
    err_audio = []
    err_slides = []
    err_transcription = []
    vid_path = f'{SOURCE_PATH}/{vid_file}'
    aud_path = f'{FOLDER_PATH}/audio/{".".join(vid_file.split(".")[:-1])}.flac'
    if aud_path.split("/")[-1] not in os.listdir(f"{FOLDER_PATH}/audio/"):
        try:
            vid_2_flac(vid_path, aud_path)
        except:
            err_audio.append((vid_path, aud_path))
    slideshow_path = f'{FOLDER_PATH}/slideshows/{".".join(vid_file.split(".")[:-1])}_slideshow.json'
    if slideshow_path.split("/")[-1] not in os.listdir(f"{FOLDER_PATH}/slideshows/"):
        try:
            hashes = compute_batch_hashes(vid_path)
            threshold = compute_threshold(hashes)
            slideshow = get_slides(vid_path, hashes, threshold)
            with open(slideshow_path,"w") as f:
                f.write(json.dumps(slideshow)) 
        except:
            err_slides.append(vid_path)
    transc_path = f'{FOLDER_PATH}/transcripts/{".".join(vid_file.split(".")[:-1])}.json'
    if os.path.isfile(aud_path) and not os.path.isfile(transc_path):
        try:
            result = transcribe(transcription_model, aud_path)
            with open(transc_path, "w") as f:
                f.write(json.dumps(result))
        except:
            err_transcription.append(aud_path)
    else:
        err_transcription.append(aud_path)
    return err_audio, err_slides, err_transcription

if __name__ == "__main__":
    transcription_model = whisper.load_model("small")
    # with open("./record_graph_construction/dl_status.json", "r") as f:
    #     dl_status = json.load(f)
    with open("./record_graph_construction/dl_status_local.json", "r") as f:
        dl_status_local = json.load(f)
    for k in random.sample(dl_status_local.keys(), len(dl_status_local.keys())):
        print(k, end=" ... ")
        if not dl_status_local[k]:
            os.system(f"wget -P ./videos/ {k}")
            err_audio, err_slides, err_transcription = process_video(transcription_model, k.split("/")[-1])
            if (len([*err_audio, *err_slides, *err_transcription])) > 0:
                print("error")
                #print(err_audio, err_slides, err_transcription)
            else:
                print("ok")
            dl_status_local[k] = True
            with open("./record_graph_construction/dl_status_local.json", "w") as f:
                f.write(json.dump(dl_status_local))
            os.system(f"rm ./videos/{k.split('/')[-1]}")

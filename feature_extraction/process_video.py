import os
import json
from tqdm import tqdm

from feature_extraction.audio_extraction import vid_2_flac
from feature_extraction.slide_extraction import compute_batch_hashes, compute_threshold, get_slides
from feature_extraction.transcript_extraction import transcribe
from feature_extraction.utils import get_params

extraction_params = get_params("./record_graph_construction/extraction_params.json")
SOURCE_PATH = extraction_params["source_path"]
FOLDER_PATH = extraction_params["folder_path"]
IMG_FOLDER_PATH = extraction_params["img_folder_path"]

# Extract video

def process_video(vid_file, pbar, verbose=True):
    """
    process the video by
      * extracting audio to a flac file (saved @ " {SOURCE_PATH}/audio/{video_name}.flac)
      * extracting the slides (to jpg files and metadata
        with timestamps @ {SOURCE_PATH}/metadata/{video_name}_slideshow.json)
      * extracting the trancript from the flac file (saved @ {SOURCE_PATH}/transcripts/{video_name}.json)

    """
    err_audio = []
    err_slides = []
    err_transcription = []
    vid_path = f'{SOURCE_PATH}/{vid_file}'
    #extract audio to flac file
    if verbose:
        pbar.set_description("Extracting audio to flac file")
    aud_path = f'{FOLDER_PATH}/audio/{".".join(vid_file.split(".")[:-1])}.flac'
    if aud_path.split("/")[-1] not in os.listdir(f"{FOLDER_PATH}/audio/"):
        #print(aud_path.split("/")[-1])
        try:
            vid_2_flac(vid_path, aud_path)
        except:
            err_audio.append((vid_path, aud_path))
    #extract slides
    if verbose:
        pbar.set_description("Extracting slides")
    path = f'{FOLDER_PATH}/metadata/{".".join(vid_file.split(".")[:-1])}_slideshow.json'
    if path.split("/")[-1] not in os.listdir(f"{FOLDER_PATH}/metadata/"):
        try:
            hashes = compute_batch_hashes(vid_path)
            # plot_distance_per_patch(hashes)
            threshold = compute_threshold(hashes)
            slideshow = get_slides(vid_path, hashes, threshold)
            with open(path,"w") as f:
                f.write(json.dumps(slideshow)) 
        except:
            err_slides.append(vid_path)
    #extract transcripts
    if verbose:
        pbar.set_description("Generating transcripts")
    transc_path = f'{FOLDER_PATH}/transcripts/{".".join(vid_file.split(".")[:-1])}.json'
    if os.path.isfile(aud_path) and not os.path.isfile(transc_path):
        try:
            result = transcribe(aud_path)
            with open(transc_path, "w") as f:
                f.write(json.dumps(result))
        except:
            err_transcription.append(aud_path)
    else:
        pass
    return err_audio, err_slides, err_transcription

def process_all_videos(verbose=True):
    pbar = tqdm(sorted(os.listdir(SOURCE_PATH)))
    for vid_file in pbar:
        err_audio, err_slides, err_transcription = process_video(vid_file, pbar, verbose)
        if verbose and (len(*err_audio, *err_slides, *err_transcription)) > 0:
            print(err_audio, err_slides, err_transcription)

if __name__ == "__main__":
    process_all_videos()
import whisper
from utils import get_params
import warnings
import os
import json

extraction_params = get_params("./record_graph_construction/extraction_params.json")
SOURCE_PATH = extraction_params["source_path"]
FOLDER_PATH = extraction_params["folder_path"]


model = whisper.load_model("small")

def transcribe(audio):
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    return model.transcribe(audio)

def transcribe_all():
    err_transcription = []
    for vid_file in os.listdir(SOURCE_PATH):
        aud_path = f'{FOLDER_PATH}/audio/{".".join(vid_file.split(".")[:-1])}.flac'
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
    return err_transcription

if __name__ == "__main__":
    print(transcribe_all())
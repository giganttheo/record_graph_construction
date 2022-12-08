from tqdm import tqdm
import json
import os

from process_video import process_video

if __name__ == "__main__":
    with open("./record_graph_construction/dl_status.json", "r") as f:
        dl_status = json.load(f)
    with open("./record_graph_construction/dl_status_local.json", "r") as f:
        dl_status_local = json.load(f)
    for k in dl_status.keys():
        print(k)
        if not dl_status[k] and not dl_status_local[k]:
            os.system(f"wget -O ./videos/ {k}")
            dl_status_local[k] = True
            with open("./dl_status_local.json", "w") as f:
                f.dump(dl_status_local, f)
            err_audio, err_slides, err_transcription = process_video(k.split("/")[-1], None, False)
            if (len(*err_audio, *err_slides, *err_transcription)) > 0:
                print(err_audio, err_slides, err_transcription)
            dl_status_local[k] = True
            with open("./record_graph_construction/dl_status_local.json", "w") as f:
                json.dump(dl_status_local, f)
            os.system(f"rm ./videos/{k.split('/')[-1]}")

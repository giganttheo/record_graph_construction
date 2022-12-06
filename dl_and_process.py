from tqdm import tqdm
import json
import os

from file_check import check_folder
from process_video import process_video


if __name__ == "__main__":
    with open("./record_graph_construction/dl_status.json", "r") as f:
        dl_status = json.load(f)
    with open("./record_graph_construction/dl_status_local.json", "r") as f:
        dl_status_local = json.load(f)
    for k in tqdm(dl_status.keys()):
        if not dl_status[k] and not dl_status_local[k]:
            os.system(f"wget -O ./videos/ {k}")
            dl_status_local[k] = True
            with open("./dl_status_local.json", "w") as f:
                f.dump(dl_status_local, f)
            process_video(k.split("/")[-1], None, False)
            os.system(f"rm ./videos/{k.split('/')[-1]}")

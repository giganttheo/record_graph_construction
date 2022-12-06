from tqdm import tqdm
import json
import os


if __name__ == "__main__":
    with open("./dl_status.json", "r") as f:
        dl_status = json.load(f)
    for k in dl_status.keys():
        if not dl_status[k]:
            os.system(f"wget {k}")
            break

import os
from tqdm import tqdm
import json
from record_graph_construction.utils import get_params

extraction_params = get_params("./record_graph_construction/extraction_params.json")
SOURCE_PATH = extraction_params["source_path"]
FOLDER_PATH = extraction_params["folder_path"]
IMG_FOLDER_PATH = extraction_params["img_folder_path"]

#check whether a file is already downloaded

def check_folder(file, folder):
    file_path = os.path.join(folder, file)
    return os.path.isfile(file_path)

def check_license(dataset, i_col):
    return "CC Attribution" in dataset[i_col]["license"].split(":")[0]


if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("gigant/tib_complete_metadata")["train"]
    dl_status = {}
    for i in tqdm(dataset):
        if check_license(dataset, i):
            url_vid = dataset[i]["url_vid"]
            dl_status[url_vid] = check_folder(url_vid.split("/")[-1], SOURCE_PATH)
    with open("dl_status.json", "w") as f:
        json.dump(dl_status, f)
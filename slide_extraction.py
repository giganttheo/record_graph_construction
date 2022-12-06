import sys
sys.path.append('./record_graph_construction/imagehash_jax')
import imagehash_jax.imagehash_jax as imagehash_jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import statistics
from decord import VideoReader
from decord import cpu
import jax
import os
import json
from tqdm import tqdm
from utils import get_params

extraction_params = get_params("./record_graph_construction/extraction_params.json")
BATCH_SIZE = extraction_params["batch_size"]
DOWNSAMPLE = extraction_params["downsampling"]
SOURCE_PATH = extraction_params["source_path"]
FOLDER_PATH = extraction_params["folder_path"]
IMG_FOLDER_PATH = extraction_params["img_folder_path"]

@jax.jit
@jax.vmap
def get_patches(img):
    shape = img.shape
    h, w = shape[0]//3, shape[1]//3
    patches=[]
    for i in range(3):
        for j in range(3):
            patch = img[i*h:((i+1)*h), j*w:((j+1)*w), :]
            patches.append(patch)
    return jnp.stack(patches)


def binary_array_to_hex(arr):
	"""
	Function to make a hex string out of a binary array.
	"""
	bit_string = ''.join(str(b) for b in 1 * arr.flatten())
	width = int(jnp.ceil(len(bit_string) / 4))
	return '{:0>{width}x}'.format(int(bit_string, 2), width=width)

def compute_batch_hashes(vid_path):
  kwargs={"width": 3*64, "height":3*64}
  vr = VideoReader(vid_path, ctx=cpu(0), **kwargs)
  hashes = []
  h_prev = None
  batch = []
  for i in range(0, len(vr), DOWNSAMPLE * BATCH_SIZE):
      ids = [id for id in range(i, min(i + DOWNSAMPLE * BATCH_SIZE, len(vr)), DOWNSAMPLE)]
      vr.seek(0)
      batch = jnp.array((vr.get_batch(ids).asnumpy()))
      batch = get_patches(batch) #batch of patches
      batch_h =  jax.vmap(imagehash_jax.batch_phash)(batch)
      for i in range(len(ids)):
        h = batch_h[i]
        if h_prev == None:
          h_prev=h
        hashes.append({"frame_id":ids[i], "hash": [binary_array_to_hex(h_patch) for h_patch in h], "distance": [int(imagehash_jax.hash_dist(h[i], h_prev[i])) for i in range(h.shape[0])]})
        h_prev = h
  return hashes

def plot_distance_per_patch(hashes):
  ids = [h["frame_id"] for h in hashes]
  for patch in range(9):
    plt.figure(figsize=(16,8))
    distance = [h["distance"][patch] for h in hashes]
    plt.plot(ids, distance, "." , label=f"Patch {patch}")
    plt.legend()
    plt.show()
    
def stats_per_patch(hashes):
  ids = [h["frame_id"] for h in hashes]
  for patch in range(9):
    distance = [h["distance"][patch] for h in hashes]
    #print(f"Patch {patch}: min = {min(distance)}, max = {max(distance)}, median = {statistics.median(distance)}, mean = {statistics.mean(distance)}")
    

def compute_threshold(hashes):
  min_length = 24 * 28
  ids = [h["frame_id"] for h in hashes]
  distances_ = [[h["distance"][patch] for h in hashes] for patch in range(9)]
  means = [statistics.mean(distance) for distance in distances_]
  if np.count_nonzero([m > 1 for m in means]) <= 5:
    ignore = [m > 1 for m in means]
  else:
    ignore = [False]*9
  thrs_ = sorted(list(set([d for distances in distances_ for d in distances])),reverse=True)
  best = thrs_[0] - 1
  for threshold in thrs_[1:-1]:
    durations = []
    i_start=0
    for i, h in enumerate(hashes):
      for patch_i in range(9):
        cond = np.any([h["distance"][i] > threshold and not ignore[i] for i in range(9)])
        if cond and (hashes[i-1]["frame_id"] -  hashes[i_start]["frame_id"]) > 12:
          durations.append(hashes[i-1]["frame_id"] - hashes[i_start]["frame_id"])
          i_start=i
    # number of frames should be < length of the video / 10 (1 frame per 40 second in average)
    # OR the median duration between two keyframes should be > 10 secs
    if len(durations) > 0:
      pass
      #print(f"Seuil: {threshold}; num kf = {len(durations)} / {(len(hashes) * DOWNSAMPLE / 24)} * ?, mean = {statistics.mean(durations)}, median = {statistics.median(durations)}")
    if len(durations) < (len(hashes) * DOWNSAMPLE / 24) / 40 or statistics.median(durations) > min_length :
      best = threshold
    else:
      break
  return best

def get_slides(vid_path, hashes, threshold):
    min_length = 24 * 1.5
    vr = VideoReader(vid_path, ctx=cpu(0))
    slideshow = []
    i_start = 0
    id_start = 0
    distances_ = [[h["distance"][patch] for h in hashes] for patch in range(9)]
    means = [statistics.mean(distance) for distance in distances_]
    if np.count_nonzero([m > 1 for m in means]) <= 5:
        ignore = [m > 1 for m in means]
    else:
        ignore = [False]*9
    for i, h in enumerate(hashes):
        cond = np.any([h["distance"][i] > threshold and not ignore[i] for i in range(9)])
        if cond and hashes[i-1]["frame_id"] - hashes[i_start]["frame_id"] >= min_length:
            path=f'{IMG_FOLDER_PATH}/{vid_path.split("/")[-1].split(".")[0]}_{id_start}_{h["frame_id"]-1}.png'
            Image.fromarray(vr[hashes[i-1]["frame_id"]].asnumpy()).save(path)
            slideshow.append({"slide": path, "frames": (id_start, h["frame_id"]-1), "timestamp": (float(vr.get_frame_timestamp(id_start)[0]), float(vr.get_frame_timestamp(h["frame_id"])[0]))})
            id_start=h["frame_id"]
            i_start=i
    path=f'{IMG_FOLDER_PATH}/{vid_path.split("/")[-1].split(".")[0]}_{id_start}_{len(vr)-1}.png'
    Image.fromarray(vr[-1].asnumpy()).save(path)
    slideshow.append({"slide": path, "frames": (id_start, len(vr)-1), "timestamp":(float(vr.get_frame_timestamp(id_start)[0]), float(vr.get_frame_timestamp(len(vr)-1)[1]))})
    return slideshow


def extract_all_slides():
    err_slides = []

    for vid_file in tqdm(os.listdir(SOURCE_PATH)):
        vid_path = f'{SOURCE_PATH}/{vid_file}'
        path = f'{FOLDER_PATH}/metadata/{".".join(vid_file.split(".")[:-1])}_slideshow.json'
        if path.split("/")[-1] not in os.listdir(f"{FOLDER_PATH}/metadata/"):
            #try:
            hashes = compute_batch_hashes(vid_path)
            # plot_distance_per_patch(hashes)
            threshold = compute_threshold(hashes)
            slideshow = get_slides(vid_path, hashes, threshold)
            with open(path,"w") as f:
                f.write(json.dumps(slideshow)) 
            # except:
            #     err_slides.append(vid_path)
    return err_slides

if __name__ == "__main__":
    print(extract_all_slides())
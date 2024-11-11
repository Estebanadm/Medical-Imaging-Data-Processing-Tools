import cv2 
from utils import * 
from pathlib import Path 
import os 
from tqdm import tqdm 

STARTING_POINT="../Data_collection-MD1"
current_dir = Path(STARTING_POINT)
all_mp4 = [str(x) for x in current_dir.glob("**/*.mp4")]
# print(all_mp4)

for filename in tqdm(all_mp4):
    video_path = filename
    saved_dir = "".join(video_path.split(".")[:3])
    saved_dir = '..'+'/'.join(saved_dir.split('/')[:-1])+"/"+str(saved_dir.split('/')[-1])
    print(saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("{}/frame{}.jpg".format(saved_dir, str(count).zfill(6)), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1


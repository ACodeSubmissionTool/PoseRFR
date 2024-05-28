import sys
sys.path.append(".")
import os
from argparse import ArgumentParser

from utils.file import list_image_files


parser = ArgumentParser()
parser.add_argument("--img_folder", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)

args = parser.parse_args()

files = list_image_files(
    args.img_folder, exts=(".jpg", ".png", ".jpeg"),
    log_progress=True, log_every_n_files=10000
)

print(f"find {len(files)} images in {args.img_folder}")


with open(args.save_path,'w') as file:
    for f in files:
        file.write(f+'\n')


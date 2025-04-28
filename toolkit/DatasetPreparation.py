import os
import sys

import cv2
import numpy as np
from tqdm import tqdm
import lmdb
import yaml
from PIL import Image
from os import path as osp
from multiprocessing import Pool
from utils import scandir
from LMDB import create_lmdb

# Code from https://github.com/jwgdmkj/LMLT/blob/main/scripts/data_preparation

def rename_subimages(folder_path):
    img_list = list(scandir(folder_path, recursive=True, full_path=False))
    for idx, original_name in enumerate(tqdm(img_list, unit='image', desc='Ordering Directory.')):
        new_name = f"{idx:08d}.png"
        src = osp.join(folder_path, original_name)
        dst = osp.join(folder_path, new_name)
        os.rename(src, dst)

def extract_subimages(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        return

    img_list = list(scandir(input_folder, recursive=True, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    rename_subimages(save_folder)
    print('All processes done.')


def worker(path, opt):
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.yaml')
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print("Config loaded successfully.")

    path = os.path.join(config["data"]["data_path"], config["data"]["training_data_name"])

    if config["tools"]["use_sub_images"]:
        #process_to_sub_images(path, path + "_sub", config["tools"]["sub_img_col"], config["tools"]["sub_img_row"])
        opt = {}
        opt['n_thread'] = 20
        opt['compression_level'] = 3
        opt['input_folder'] = path
        opt['save_folder'] = path + "_sub"
        opt['crop_size'] = 480
        opt['step'] = 240
        opt['thresh_size'] = 10
        extract_subimages(opt)
        path += "_sub"

    if config["tools"]["use_lmdb"]:
        create_lmdb(path, path+"_lmdb")


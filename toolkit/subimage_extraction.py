import os
import cv2
import numpy as np
from tqdm import tqdm
from os import path as osp
from multiprocessing import Pool
from .utils import scandir
from .utils import format_image_files

def extract_subimages(input_folder, save_folder, opt):
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        return

    img_list = list(scandir(input_folder, recursive=True, full_path=True))
    pbar = tqdm(total=len(img_list), unit='image', desc='Extracting Sub-Images')
    pool = Pool(opt['n_threads'])
    for src in img_list:
        pool.apply_async(extract_subimages_worker, args=(src, save_folder, opt,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    format_image_files(save_folder)


def extract_subimages_worker(src, save_folder, opt):
    crop_size_h = opt['crop_size']
    crop_size_w = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(src))
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]

    if h < crop_size_h:
        crop_size_h = h
    if w < crop_size_w:
        crop_size_w = w

    h_space = _calculate_positions(h, crop_size_h, step, thresh_size)
    w_space = _calculate_positions(w, crop_size_w, step, thresh_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size_h, y:y + crop_size_w, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(save_folder, f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])


def _calculate_positions(dim_size, crop_size, step, thresh_size):
    positions = np.arange(0, dim_size - crop_size + 1, step)
    if dim_size - (positions[-1] + crop_size) > thresh_size:
        if positions[-1] != (dim_size - crop_size):
            positions = np.append(positions, dim_size - crop_size)
    return positions
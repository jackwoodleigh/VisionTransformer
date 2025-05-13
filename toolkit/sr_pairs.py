import os
import cv2
from tqdm import tqdm
from os import path as osp
from multiprocessing import Pool
from .utils import scandir
from .transforms import bicubic_downscale_HWC


def create_paired_images(data_root, opt):
    img_list = list(scandir(data_root, recursive=True, full_path=True))
    if not osp.exists(osp.join(data_root, "hr")):
        os.makedirs(osp.join(data_root, "hr"))
    else:
        print(f"{osp.join(data_root, 'hr')} already exists.")

    if not osp.exists(osp.join(data_root, "lr")):
        os.makedirs(osp.join(data_root, "lr"))
    else:
        print(f"{osp.join(data_root, 'lr')} already exists.")

    pbar = tqdm(total=len(img_list), unit='image', desc='Creating Pairs')
    pool = Pool(opt['n_threads'])
    for src in img_list:
        pool.apply_async(create_paired_images_worker, args=(data_root, src, opt,), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()

def create_paired_images_worker(data_root, src, opt):
    name = osp.basename(src)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    lr_img = cv2.resize(img, (img.shape[1] // opt['scale'], img.shape[0] // opt['scale']), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(osp.join(data_root, 'lr', name), lr_img, [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    os.rename(src, osp.join(data_root, 'hr', name))

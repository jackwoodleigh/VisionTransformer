import sys
from multiprocessing import Pool, cpu_count
import cv2
import lmdb
from os import path as osp
from tqdm import tqdm
import os
from .utils import scandir

def get_keys_from_lmdb(lmdb_path):
    image_keys_list = []
    with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False) as env:
        with env.begin(write=False) as txn:
            meta_keys_bytes = {b'__len__', b'__compress_level__'}

            cursor = txn.cursor()
            for key_bytes, _ in cursor:
                if key_bytes not in meta_keys_bytes:
                    image_keys_list.append(key_bytes.decode('utf-8'))

    return sorted(image_keys_list), len(image_keys_list)

def create_lmdb_from_folder(folder_path, lmdb_path, map_size_gb=None, n_thread=None, batch_size=5000, compress_level=1):

    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        return

    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True, full_path=False)))

    if not img_path_list:
        print(f"No PNG images found in {folder_path}. Exiting.")
        return

    keys = [osp.splitext(img_file)[0] for img_file in img_path_list]

    if n_thread is None:
        n_thread = cpu_count()

    if map_size_gb is None:
        first_img_path = osp.join(folder_path, img_path_list[0])
        _, first_img_byte, _ = read_img_worker_args((first_img_path, "dummy_key", compress_level))
        data_size_per_img = len(first_img_byte)
        estimated_total_size = data_size_per_img * len(img_path_list)
        map_size = estimated_total_size * 2.5 # buffer
        print(f"Estimated data size: {estimated_total_size / (1024**3):.2f} GiB. Setting map_size to {map_size / (1024**3):.2f} GiB.")
    else:
        map_size = int(map_size_gb * 1024 * 1024 * 1024)

    with lmdb.open(lmdb_path, map_size=map_size) as env:
        with open(osp.join(lmdb_path, 'meta_info.txt'), 'w') as meta_file:
            meta_file.write(f'LMDB created from: {osp.abspath(folder_path)}\n')
            meta_file.write(f'Total images: {len(img_path_list)}\n')
            meta_file.write(f'Key format: filename (without .png extension)\n')
            meta_file.write(f'PNG Compression Level: {compress_level}\n')
            meta_file.write('--- Image List (key, original_filename, shape (h,w,c)) ---\n')

            txn = env.begin(write=True)
            txn.put(b'__len__', str(len(img_path_list)).encode('utf-8'))
            txn.put(b'__compress_level__', str(compress_level).encode('utf-8'))
            txn.commit()
            txn = env.begin(write=True)

            tasks = [(osp.join(folder_path, rel_path), key, compress_level) for rel_path, key in zip(img_path_list, keys)]

            with Pool(processes=n_thread) as pool, tqdm(total=len(img_path_list), unit='image', desc="Creating LMDB") as pbar:
                for i, (result_key, img_byte, img_shape) in enumerate(pool.imap_unordered(read_img_worker_args, tasks)):
                    if img_byte is None:
                        print(f"Warning: Failed to read/process image for key {result_key}. Skipping.")
                        meta_file.write(f'{result_key} ERROR_READING_IMAGE\n')
                        continue

                    key_byte = result_key.encode('utf-8')
                    txn.put(key_byte, img_byte)

                    h, w, c = img_shape
                    original_filename = f"{result_key}.png"
                    meta_file.write(f'{result_key} {original_filename} ({h},{w},{c})\n')

                    pbar.update(1)

                    if (i + 1) % batch_size == 0:
                        txn.commit()
                        txn = env.begin(write=True)

                txn.commit()

def read_img_worker_args(args):
    path, key, compress_level = args
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: cv2.imread returned None for path {path}")
            return key, None, (0,0,0)

        if img.ndim == 2:
            h, w = img.shape
            c = 1
        else:
            h, w, c = img.shape

        _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        return key, img_byte.tobytes(), (h, w, c)
    except Exception as e:
        print(f"Error processing image {path} for key {key}: {e}")
        return key, None, (0,0,0)


if __name__ == '__main__':
    # test
    dummy_data_folder = 'dummy_images'
    dummy_lmdb_path = 'dummy_images.lmdb'

    import shutil
    if osp.exists(dummy_data_folder):
        shutil.rmtree(dummy_data_folder)
    if osp.exists(dummy_lmdb_path):
        shutil.rmtree(dummy_lmdb_path)

    os.makedirs(dummy_data_folder, exist_ok=True)

    import numpy as np
    for i in range(100000):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(osp.join(dummy_data_folder, f'image_{i:03d}.png'), img)
    img_gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    cv2.imwrite(osp.join(dummy_data_folder, f'image_gray.png'), img_gray)

    print("Creating LMDB...")

    create_lmdb_from_folder(dummy_data_folder, dummy_lmdb_path, map_size_gb=2, n_thread=8, compress_level=1)

    print(f"\nVerifying LMDB {dummy_lmdb_path}...")
    keys, count = get_keys_from_lmdb(dummy_lmdb_path)
    print(f"Found {count} images with keys: {keys[:5]}...")

    if keys:
        with lmdb.open(dummy_lmdb_path, readonly=True, lock=False) as env:
            with env.begin(write=False) as txn:
                first_key = keys[0].encode('ascii') # or 'utf-8'
                img_bytes_from_lmdb = txn.get(first_key)
                if img_bytes_from_lmdb:
                    img_np_array = np.frombuffer(img_bytes_from_lmdb, dtype=np.uint8)
                    img_decoded = cv2.imdecode(img_np_array, cv2.IMREAD_UNCHANGED)
                    print(f"Successfully decoded image for key '{keys[0]}' with shape: {img_decoded.shape}")
                else:
                    print(f"Could not find image for key '{keys[0]}' in LMDB.")

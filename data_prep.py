import os
import yaml
from toolkit.lmdb_utils import create_lmdb_from_folder
from toolkit.utils import format_image_files
from toolkit.subimage_extraction import extract_subimages
from toolkit.sr_pairs import create_paired_images
from multiprocessing import cpu_count

if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print("Config loaded successfully.")

    opt = {
        'n_threads': cpu_count(),
        'compression_level': 3,
        'crop_size': 480,
        'step': 240,
        'thresh_size': 10,
        'scale': config["model"]["scale_factor"]
    }

    data_root = os.path.join(config["dataset"]["data_root"], config["dataset"]["training"]["file_name"])
    print("Preparing data...")
    format_image_files(data_root, flatten=True)

    if config["dataset"]["training"]["use_sub_images"]:
        extract_subimages(data_root, data_root + "_sub", opt)
        data_root += "_sub"

    if config["dataset"]["training"]["paired_images"]:
        create_paired_images(data_root, opt)

    if config["tools"]["file_backend_type"] == "LMDB":
        create_lmdb_from_folder(data_root, data_root + "_lmdb", map_size_gb=60)
        data_root += "_lmdb"

    print(f"Data Preparation Complete! Your data can be found at {data_root}")

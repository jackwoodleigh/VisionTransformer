import os
import cv2
import lmdb
import numpy as np

def bytes_to_rgb(x):
    return (cv2.imdecode(np.frombuffer(x, dtype=np.uint8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.).transpose(2, 0, 1)

class FileBackend:
    def __init__(self, backend_type, root):
        self.backend_type = backend_type
        self.root = root
        self.env = None

    def load_from_LMDB(self, img_name):
        if self.env is None:
            self.env = lmdb.open(
                self.root,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
        key = os.path.splitext(img_name)[0]
        key = str(key).encode('utf-8')
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(key)
            if img_bytes is None:
                raise KeyError(f"Key {key} not found in LMDB!")
        return img_bytes

    def load_from_disk(self, img_name):
        filepath = os.path.join(self.root, img_name)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def load(self, img_name):
        if self.backend_type == "LMDB":
            return bytes_to_rgb(self.load_from_LMDB(img_name))
        else:
            return bytes_to_rgb(self.load_from_disk(img_name))

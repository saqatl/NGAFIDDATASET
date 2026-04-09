"""
Lightweight NGAFID data loader for the PyTorch training pipeline.

Replicates the essential logic of ngafiddataset.dataset.dataset without
requiring TensorFlow, so the environment only needs PyTorch.
"""

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from compress_pickle import load
from tqdm import tqdm

NGAFID_URLS = {
    "2days": "https://zenodo.org/records/6624956/files/2days.tar.gz?download=1",
}

CHANNELS = 23
MAX_LENGTH = 4096


def _download_and_extract(name: str = "2days", destination: str = ""):
    url = NGAFID_URLS[name]
    archive = os.path.join(destination, f"{name}.tar.gz")
    data_dir = os.path.join(destination, name)

    if not os.path.exists(archive):
        print(f"Downloading {name} dataset from Zenodo …")
        response = urllib.request.urlopen(url)
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=name) as pbar:
            with open(archive, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Saved → {archive}")

    if not os.path.isdir(data_dir):
        print("Extracting …")
        with tarfile.open(archive) as tar:
            tar.extractall(destination)

    return data_dir


class NGAFIDDataset:
    """Load the NGAFID 2-days benchmark dataset into numpy arrays."""

    def __init__(self, name: str = "2days", destination: str = "."):
        data_dir = _download_and_extract(name, destination)

        self.flight_header = pd.read_csv(
            os.path.join(data_dir, "flight_header.csv"), index_col="Master Index"
        )
        self.flight_data = load(os.path.join(data_dir, "flight_data.pkl"))
        stats = pd.read_csv(os.path.join(data_dir, "stats.csv"))

        self.maxs = stats.iloc[0, 1 : CHANNELS + 1].to_numpy(dtype=np.float32)
        self.mins = stats.iloc[1, 1 : CHANNELS + 1].to_numpy(dtype=np.float32)

        unique_labels = sorted(self.flight_header["target_class"].unique())
        self._label2idx = {lab: idx for idx, lab in enumerate(unique_labels)}
        self._idx2label = {idx: lab for lab, idx in self._label2idx.items()}

        self._data_dict = self._build_data_dict()

    @property
    def num_classes(self) -> int:
        return len(self._label2idx)

    @property
    def label_map(self) -> dict:
        """Maps contiguous index -> original target_class value."""
        return dict(self._idx2label)

    def _build_data_dict(self) -> list[dict]:
        records: list[dict] = []
        for idx, row in tqdm(
            self.flight_header.iterrows(),
            total=len(self.flight_header),
            desc="Building data dictionary",
        ):
            arr = np.zeros((MAX_LENGTH, CHANNELS), dtype=np.float16)
            raw = self.flight_data[idx][-MAX_LENGTH:, :]
            arr[: raw.shape[0], :] += raw
            records.append(
                {
                    "data": arr,
                    "target_class": row["target_class"],
                    "before_after": row["before_after"],
                    "fold": row["fold"],
                }
            )
        return records

    def get_fold_data(
        self, fold: int, training: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, y_binary, y_multiclass) for the given fold.

        training=True  -> all folds except `fold`  (4/5 of data)
        training=False -> only `fold`               (1/5 of data)

        X            : float32, min-max normalised, NaN replaced with 0.
        y_binary     : int64, 0 = after maintenance, 1 = before maintenance.
        y_multiclass : int64, remapped to contiguous [0, num_classes).
        """
        if training:
            subset = [d for d in self._data_dict if d["fold"] != fold]
        else:
            subset = [d for d in self._data_dict if d["fold"] == fold]

        X = np.array([d["data"] for d in subset], dtype=np.float32)
        denom = self.maxs - self.mins
        denom[denom == 0] = 1.0
        X = (X - self.mins) / denom
        np.nan_to_num(X, copy=False)

        y_binary = np.array(
            [d["before_after"] for d in subset], dtype=np.int64
        )
        y_multi = np.array(
            [self._label2idx[d["target_class"]] for d in subset], dtype=np.int64
        )
        return X, y_binary, y_multi

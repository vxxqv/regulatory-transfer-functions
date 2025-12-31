"""Read only observation metadata from four bounded remote HDF5 releases."""

from collections import OrderedDict
import io
import json
import urllib.request

import h5py
import numpy as np
import pandas as pd

from analyses.compositionality.audit_metadata import OUT


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, block_size=262144):
        self.url, self.size, self.block_size = url, size, block_size
        self.position = 0
        self.cache = OrderedDict()
        self.bytes_transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        result = bytearray()
        while self.position < end:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                stop = min(start + self.block_size, self.size) - 1
                request = urllib.request.Request(self.url, headers={
                    "User-Agent": "regulatory-transfer-study/1.0",
                    "Range": f"bytes={start}-{stop}",
                })
                with urllib.request.urlopen(request, timeout=35) as response:
                    if response.status != 206:
                        raise ValueError("Range requests unavailable; full release was not downloaded")
                    chunk = response.read()
                assert len(chunk) == stop - start + 1
                self.cache[block] = chunk
                self.bytes_transferred += len(chunk)
                if self.bytes_transferred > 25000000:
                    raise ValueError("Metadata transfer exceeded 25 MB stopping rule")
                if len(self.cache) > 40:
                    self.cache.popitem(last=False)
            chunk = self.cache[block]
            offset = self.position % self.block_size
            piece = chunk[offset:offset + min(end - self.position, self.block_size - offset)]
            result.extend(piece)
            self.position += len(piece)
        return bytes(result)

    def readinto(self, buffer):
        result = self.read(len(buffer))
        buffer[:len(result)] = result
        return len(result)


def read_column(group, name):
    node = group[name]
    if isinstance(node, h5py.Group):
        categories = node["categories"].asstr()[:] if node["categories"].dtype.kind in "OSU" else node["categories"][:]
        codes = node["codes"][:]
        return np.where(codes >= 0, categories[np.maximum(codes, 0)], "unassigned")
    return node.asstr()[:] if node.dtype.kind in "OSU" else node[:]


def inspect_release(name):
    allowed = {"TianKampmann2019_iPSC", "TianKampmann2019_day7neuron", "AdamsonWeissman2016_GSM2406681_10X010", "SunshineHein2023"}
    if name not in allowed:
        raise ValueError("Dataset outside the metadata-only acquisition scope")
    catalog = json.loads((OUT / "scperturb_zenodo.json").read_text())
    file = next(row for row in catalog["files"] if row["key"] == name + ".h5ad")
    reader = RangeReader(file["links"]["self"], file["size"])
    report = {"dataset": name, "release_bytes": file["size"], "source_checksum": file["checksum"],
              "outcome_arrays_read": False}
    try:
        with h5py.File(reader, "r") as data:
            columns = list(data["obs"])
            report["obs_columns"] = columns
            wanted = [col for col in columns if col in {"perturbation", "perturbation_type", "batch", "ngenes"}]
            obs = pd.DataFrame({col: read_column(data["obs"], col) for col in wanted})
            report["cells"] = len(obs)
            report["metadata_levels"] = {col: {str(k): int(v) for k, v in obs[col].value_counts(dropna=False).items()}
                                         for col in wanted if obs[col].nunique() < 400}
            obs.to_csv(OUT / (name + "_obs.tsv.gz"), sep="\t", index=False)
            report["status"] = "metadata_retrieved"
    except Exception as error:
        report["status"] = "metadata_unavailable"
        report["reason"] = str(error)
    report["bytes_transferred"] = reader.bytes_transferred
    (OUT / (name + "_audit.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metadata_levels"}), flush=True)


if __name__ == "__main__":
    import sys
    for dataset in sys.argv[1:]:
        inspect_release(dataset)

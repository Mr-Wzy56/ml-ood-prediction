"""Prepare and verify all five datasets for the Pre2 experiments.

Origin servers (cs231n.stanford.edu, ufldl.stanford.edu, robots.ox.ac.uk) are
unreachable from the current AutoDL instance, so the remaining datasets are
fetched from official/community HuggingFace mirrors via hf-mirror.com and
reconstructed into the layouts expected by src/data.py:

- CIFAR-10 / CIFAR-100: original python archives copied from /root/autodl-pub
  (MD5-verified by torchvision).
- Tiny ImageNet: parquet from hf-mirror.com/datasets/zh-plus/tiny-imagenet
  (100k train images, 64x64) -> reconstructed as ImageFolder
  data/tiny-imagenet-200/train/<wnid>/*.JPEG.
- SVHN test: parquet from hf-mirror.com/datasets/ufldl-stanford/svhn
  (cropped_digits/test, 26,032 32x32 images) -> data/_parquet/svhn_test.parquet,
  loaded by SVHNParquet in src/data.py.
- DTD test: parquet from hf-mirror.com/datasets/tanganke/dtd (test split,
  official partition-1 test images) -> data/_parquet/dtd_test.parquet, loaded by
  DTDParquet in src/data.py.

Run: python -m src.prepare_data [--root data]
"""
import argparse
import json
import struct
from pathlib import Path

import pyarrow.parquet as pq

TINY_PARQUET = "tiny_train.parquet"
SVHN_PARQUET = "svhn_test.parquet"
DTD_PARQUET = "dtd_test.parquet"

SVHN_TEST_EXPECTED = 26032
DTD_TEST_EXPECTED = 1880  # official partition-1 test: 40 images x 47 classes
TINY_TRAIN_EXPECTED = 100000


def verify_cifar(root: Path):
    from torchvision import datasets
    stats = {}
    for name, cls in [("cifar10", datasets.CIFAR10), ("cifar100", datasets.CIFAR100)]:
        tr = cls(root=root, train=True, download=False)
        te = cls(root=root, train=False, download=False)
        targets = tr.targets
        per_class = {}
        for t in targets:
            per_class[int(t)] = per_class.get(int(t), 0) + 1
        stats[name] = {
            "train_pool": len(tr),
            "test": len(te),
            "num_classes": len(per_class),
            "min_per_class_train": min(per_class.values()),
            "md5_verified": True,
        }
        print(f"[cifar] {name}: train {len(tr)} (min/class {min(per_class.values())}), test {len(te)}")
    return stats


def tiny_label_names(parquet_path: Path):
    """Read ClassLabel names from parquet HF metadata, else fall back to tiny_wnids.json."""
    import json as _json
    meta = pq.ParquetFile(parquet_path).schema_arrow.metadata
    if meta and b"huggingface" in meta:
        hf = _json.loads(meta[b"huggingface"].decode())
        return hf["info"]["features"]["label"]["names"]
    fallback = parquet_path.parent / "tiny_wnids.json"
    return _json.loads(fallback.read_text())


def reconstruct_tinyimagenet(parquet_path: Path, out_root: Path):
    import io
    from PIL import Image

    out_root = out_root / "tiny-imagenet-200" / "train"
    marker = out_root / ".reconstruction_done"
    if marker.exists():
        print("[tiny] already reconstructed, skipping")
        return
    table = pq.read_table(parquet_path)
    n = table.num_rows
    names = tiny_label_names(parquet_path)
    print(f"[tiny] decoding {n} images into {len(names)} wnid folders under {out_root}")
    images = table.column("img" if "img" in table.column_names else table.column_names[0]).to_pylist()
    labels = table.column("label").to_pylist()
    assert len(names) == max(labels) + 1, "label index outside class names"
    counts = {}
    for i, (img, label) in enumerate(zip(images, labels)):
        wnid = names[label]
        d = out_root / wnid / "images"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{wnid}_{i:06d}.JPEG"
        if not path.exists():
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(path, "JPEG", quality=95)
        counts[wnid] = counts.get(wnid, 0) + 1
    total = sum(counts.values())
    per_class = {c: counts.get(c, 0) for c in names}
    assert total == TINY_TRAIN_EXPECTED, f"expected {TINY_TRAIN_EXPECTED}, got {total}"
    assert min(per_class.values()) == 500, f"expected 500 per class, got min {min(per_class.values())}"
    marker.write_text(json.dumps({"total": total, "per_class": per_class}, indent=2))
    print(f"[tiny] OK: {total} images, 200 classes x 500")


def verify_ood_parquets(root: Path):
    import io
    from collections import Counter
    from PIL import Image

    pq_dir = root / "_parquet"
    stats = {}
    # SVHN test
    f = pq.ParquetFile(pq_dir / SVHN_PARQUET)
    assert f.metadata.num_rows == SVHN_TEST_EXPECTED, f"svhn rows {f.metadata.num_rows}"
    img0 = f.read_row_group(0).column("image")[0].as_py()
    im = Image.open(io.BytesIO(img0["bytes"]))
    stats["svhn_test"] = {"rows": f.metadata.num_rows, "image_size": im.size, "format": im.format}
    print(f"[ood] svhn test: {f.metadata.num_rows} images, size {im.size} {im.format}")

    # DTD test
    f = pq.ParquetFile(pq_dir / DTD_PARQUET)
    n = f.metadata.num_rows
    assert n == DTD_TEST_EXPECTED, f"dtd rows {n} != expected {DTD_TEST_EXPECTED}"
    labels = f.read(columns=["label"]).column("label").to_pylist()
    meta = f.schema_arrow.metadata
    import json as _json
    hf = _json.loads(meta[b"huggingface"].decode())
    names = hf["info"]["features"]["label"]["names"]
    per_class = Counter(names[l] for l in labels)
    img0 = f.read_row_group(0).column("image")[0].as_py()
    im = Image.open(io.BytesIO(img0["bytes"]))
    stats["dtd_test"] = {
        "rows": n,
        "num_classes": len(names),
        "per_class": dict(sorted(per_class.items())),
        "image_size_sample": im.size,
        "format": im.format,
    }
    assert len(names) == 47 and min(per_class.values()) == 40 and max(per_class.values()) == 40, \
        "DTD partition-1 test should be 40 images x 47 classes"
    print(f"[ood] dtd test: {n} images, 47 classes x 40, sample size {im.size} {im.format}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    a = p.parse_args()
    root = Path(a.root)
    pq_dir = root / "_parquet"

    report = {"cifar": verify_cifar(root)}
    reconstruct_tinyimagenet(pq_dir / TINY_PARQUET, root)
    report["ood"] = verify_ood_parquets(root)

    out = Path("runs") / "data_preparation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"summary written to {out}")


if __name__ == "__main__":
    main()


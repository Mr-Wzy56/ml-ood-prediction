"""Download the project's five core datasets and bundle them for transfer."""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from torchvision import datasets

TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_tinyimagenet(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "tiny-imagenet-200.zip"
    target = root / "tiny-imagenet-200"
    if target.is_dir() and (target / "train").is_dir():
        print("Tiny ImageNet already extracted")
        return
    if not archive.exists() or md5(archive) != TINY_MD5:
        print(f"Downloading {TINY_URL}")
        urllib.request.urlretrieve(TINY_URL, archive)
    if md5(archive) != TINY_MD5:
        raise RuntimeError("Tiny ImageNet MD5 mismatch; remove the archive and retry")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--bundle", default="datasets_bundle.zip")
    a = p.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    print("Downloading CIFAR-10 and CIFAR-100")
    datasets.CIFAR10(root=root, train=True, download=True)
    datasets.CIFAR10(root=root, train=False, download=True)
    datasets.CIFAR100(root=root, train=True, download=True)
    datasets.CIFAR100(root=root, train=False, download=True)
    print("Downloading SVHN")
    datasets.SVHN(root=root, split="train", download=True)
    datasets.SVHN(root=root, split="test", download=True)
    print("Downloading DTD")
    datasets.DTD(root=root, split="train", download=True)
    datasets.DTD(root=root, split="val", download=True)
    datasets.DTD(root=root, split="test", download=True)
    print("Downloading and extracting Tiny ImageNet")
    download_tinyimagenet(root)
    bundle = Path(a.bundle)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating {bundle} (this may take a while)")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(root.parent))
    print(f"Done: {bundle.resolve()} ({bundle.stat().st_size / (1024**3):.2f} GiB)")

if __name__ == "__main__":
    main()


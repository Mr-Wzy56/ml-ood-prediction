from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

def transforms_for(train=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)

def id_loaders(name="cifar10", root="data", batch_size=128, val_fraction=0.1, seed=42, workers=4):
    cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    base = cls(root=root, train=True, download=True, transform=transforms_for(False))
    n_val = int(len(base) * val_fraction)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(base), generator=gen).tolist()
    train_idx, val_idx = perm[n_val:], perm[:n_val]
    train_ds = cls(root=root, train=True, download=False, transform=transforms_for(True))
    val_ds = Subset(base, val_idx)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test = cls(root=root, train=False, download=True, transform=transforms_for(False))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def tinyimagenet_outliers(root, batch_size=128, workers=4, exclude_file=None):
    """Load Tiny ImageNet train images, optionally excluding class names/wnids listed one per line."""
    from pathlib import Path
    root = Path(root)
    train_root = root / "train" if (root / "train").is_dir() else root
    ds = datasets.ImageFolder(train_root, transform=transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    if exclude_file and Path(exclude_file).exists():
        excluded = {x.strip() for x in Path(exclude_file).read_text().splitlines() if x.strip() and not x.startswith("#")}
        kept = [i for i, (_, label) in enumerate(ds.samples) if ds.classes[label] not in excluded]
        ds = torch.utils.data.Subset(ds, kept)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

class _ParquetImageDataset(torch.utils.data.Dataset):
    """Base for OOD sets stored as HuggingFace parquet (image bytes + label).

    Used because the origin servers (ufldl.stanford.edu / robots.ox.ac.uk) are
    unreachable from the current instance; files come from official mirrors via
    hf-mirror.com. Labels are kept but never used for OOD thresholding.
    """

    def __init__(self, parquet_path, transform=None):
        import pyarrow.parquet as pq
        self.path = Path(parquet_path)
        self.table = pq.read_table(self.path)
        self.transform = transform
        meta = pq.ParquetFile(self.path).schema_arrow.metadata
        import json
        hf = json.loads(meta[b"huggingface"].decode())
        self.classes = hf["info"]["features"]["label"]["names"]

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        import io
        from PIL import Image
        row = self.table.slice(int(idx), 1).to_pydict()
        img = Image.open(io.BytesIO(row["image"][0]["bytes"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"][0])


class SVHNParquet(_ParquetImageDataset):
    """SVHN cropped_digits test split (26,032 32x32 images), official HF mirror."""

    def __init__(self, root="data", transform=None):
        super().__init__(Path(root) / "_parquet" / "svhn_test.parquet", transform)


class DTDParquet(_ParquetImageDataset):
    """DTD test split (official partition-1 test: 40x47=1880 texture images)."""

    def __init__(self, root="data", transform=None):
        super().__init__(Path(root) / "_parquet" / "dtd_test.parquet", transform)


def ood_loaders(root="data", batch_size=128, workers=4):
    tf = transforms_for(False)
    sets = {
        "svhn": SVHNParquet(root=root, transform=tf),
        "dtd": DTDParquet(root=root, transform=transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])),
    }
    return {k: DataLoader(v, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True) for k,v in sets.items()}


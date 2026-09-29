import io
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_paths(directory):
    paths = sorted(path for path in Path(directory).iterdir() if path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {directory}")
    return paths


def jpeg_pillow(image, quality, subsampling="default"):
    buffer = io.BytesIO()
    options = {"format": "JPEG", "quality": int(quality)}
    if subsampling == "444":
        options["subsampling"] = 0
    image.save(buffer, **options)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


class BlindJPEGDataset(Dataset):
    def __init__(self, gt_dir, crop_size=128, qf_min=10, qf_max=70,
                 qf_step=10, discrete_epochs=1000, subsampling="444"):
        self.paths = image_paths(gt_dir)
        self.crop_size = crop_size
        self.qf_min = qf_min
        self.qf_max = qf_max
        self.qf_step = qf_step
        self.discrete_epochs = discrete_epochs
        self.subsampling = subsampling
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as source:
            gt = source.convert("RGB")
        width, height = gt.size
        size = self.crop_size
        if min(width, height) < size:
            raise ValueError(f"Image smaller than crop size {size}: {path}")
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
        gt = TF.crop(gt, top, left, size, size)
        if random.random() < 0.5:
            gt = TF.hflip(gt)
        if random.random() < 0.5:
            gt = TF.vflip(gt)
        gt = TF.rotate(gt, 90 * random.randint(0, 3))
        if self.epoch <= self.discrete_epochs:
            qf = random.choice(range(self.qf_min, self.qf_max + 1, self.qf_step))
        else:
            qf = random.randint(self.qf_min, self.qf_max)
        lq = jpeg_pillow(gt, qf, self.subsampling)
        return TF.to_tensor(lq), TF.to_tensor(gt)

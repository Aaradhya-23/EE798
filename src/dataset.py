"""PASCAL VOC 2007 with image-level labels and bounding boxes (via torchvision's VOCDetection)."""
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection

from .utils import IMAGENET_MEAN, IMAGENET_STD

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]


class VOCLocalization(Dataset):
    """Each item: normalised image (size x size), 20-d multi-hot labels, boxes [(cls,x1,y1,x2,y2)].

    Images are resized (anisotropically) to size x size so that boxes only need to be rescaled.
    'difficult' objects are ignored for labels and boxes.
    """

    def __init__(self, root, image_set="trainval", size=448, indices=None, train=False, download=True):
        self.voc = VOCDetection(root, year="2007", image_set=image_set, download=download)
        self.indices = list(range(len(self.voc))) if indices is None else list(indices)
        self.size, self.train = size, train
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, target = self.voc[idx]
        ann = target["annotation"]
        W, H = img.size
        objs = ann["object"] if isinstance(ann["object"], list) else [ann["object"]]
        labels, boxes = torch.zeros(len(VOC_CLASSES)), []
        sx, sy = self.size / W, self.size / H
        for o in objs:
            if int(o["difficult"]):
                continue
            c, bb = VOC_CLASSES.index(o["name"]), o["bndbox"]
            labels[c] = 1
            boxes.append((c, (float(bb["xmin"]) - 1) * sx, (float(bb["ymin"]) - 1) * sy,
                          float(bb["xmax"]) * sx, float(bb["ymax"]) * sy))
        img = img.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        if self.train and torch.rand(1).item() < 0.5:      # horizontal flip (boxes unused in training)
            img = ImageOps.mirror(img)
        return {"image": self.tf(img), "labels": labels, "boxes": boxes,
                "image_id": ann["filename"], "index": idx}


def collate(batch):
    return {"image": torch.stack([b["image"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "boxes": [b["boxes"] for b in batch],
            "image_id": [b["image_id"] for b in batch],
            "index": [b["index"] for b in batch]}

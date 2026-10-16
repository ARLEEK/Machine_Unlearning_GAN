import os
import requests
import zipfile
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms


COCO_URLS = {
    "train_images": (
        "http://images.cocodataset.org/zips/train2017.zip",
        "train2017.zip",
    ),
    "val_images": (
        "http://images.cocodataset.org/zips/val2017.zip",
        "val2017.zip",
    ),
    "annotations": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations_trainval2017.zip",
    ),
}


def download_file(url, destination):
    if os.path.exists(destination):
        print(f"[INFO] Already exists: {destination}")
        return

    print(f"[INFO] Downloading: {url}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file, tqdm(
        desc=destination,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def extract_zip(path, extract_to):
    folder = os.path.splitext(os.path.basename(path))[0]
    target = os.path.join(extract_to, folder)

    if os.path.exists(target):
        print(f"[INFO] Already extracted: {target}")
        return

    print(f"[INFO] Extracting {path}")
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(extract_to)


def ensure_coco_exists(root="./data/coco"):
    os.makedirs(root, exist_ok=True)

    for name, (url, zip_name) in COCO_URLS.items():
        zip_path = os.path.join(root, zip_name)

        # download the zip
        download_file(url, zip_path)

        # extract folder only if missing
        extract_zip(zip_path, root)


def load_coco_images(
    root="./data/coco",
    image_set="train2017",
    batch_size=128,
    shuffle=True,
    num_workers=4,
    image_size=64,
):

    ensure_coco_exists(root)

    img_dir = os.path.join(root, image_set)
    ann_file = os.path.join(root, "annotations", f"instances_{image_set}.json")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"COCO images not found: {img_dir}")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"COCO annotation file missing: {ann_file}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = CocoDetection(img_dir, ann_file, transform=transform)

    # Convert (image, annotations)
    dataset = [(img, 0) for img, _ in dataset]

    print(f"[INFO] Loaded {len(dataset)} COCO images from {image_set}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
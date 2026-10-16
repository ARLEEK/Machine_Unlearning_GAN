import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CIFAR-10 class indices
CIFAR10_CLASSES = {
    "airplane": 0,
    "dog": 5,
    "train": 9
}

BATCH_SIZE = 128

def load_cifar10_filtered(root="./data", batch_size=BATCH_SIZE, classes=None, shuffle=True):
    """
    Loads CIFAR-10 but keeps only the classes specified.

    Args:
        root (str): Directory where data will be downloaded.
        batch_size (int): Batch size for training.
        classes (list): List of class names, e.g. ["airplane", "dog"].
        shuffle (bool): Shuffle training data.

    Returns:
        DataLoader: PyTorch dataloader for filtered dataset.
    """

    if classes is None:
        raise ValueError("You must pass a list of classes, e.g., ['airplane', 'dog']")

    # Transform for GAN training
    transform = transforms.Compose([
        transforms.Resize(64),                # DCGAN resolution
        transforms.ToTensor(),
        # Scale to [-1, 1]
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    # Download CIFAR-10
    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    # Map class names
    target_labels = [CIFAR10_CLASSES[c] for c in classes]

    # Keep only those samples
    indices = [i for i, (_, y) in enumerate(dataset) if y in target_labels]

    filtered_dataset = Subset(dataset, indices)

    print(f"[INFO] Loaded {len(filtered_dataset)} images for classes: {classes}")

    return DataLoader(filtered_dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # loading all classes
    loader = load_cifar10_filtered(classes=["airplane", "dog", "train"])
    print("Batches:", len(loader))
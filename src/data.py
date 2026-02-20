import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import Config


def prepare_dataset():
    if os.path.exists(Config.DATA_DIR):
        print("Dataset already prepared.")
        return

    print("Preparing dataset...")

    random.seed(Config.SEED)

    classes = os.listdir(Config.ORIGINAL_DATA_DIR)

    for split in ["train", "test"]:
        for cls in classes:
            os.makedirs(
                os.path.join(Config.DATA_DIR, split, cls),
                exist_ok=True
            )

    for cls in classes:
        class_path = os.path.join(Config.ORIGINAL_DATA_DIR, cls)

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * Config.TRAIN_SPLIT)

        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for img in train_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(Config.DATA_DIR, "train", cls, img)
            )

        for img in test_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(Config.DATA_DIR, "test", cls, img)
            )

    print("Dataset prepared successfully.")


def get_dataloaders():
    prepare_dataset()

    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(Config.DATA_DIR, "train"),
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(Config.DATA_DIR, "test"),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader

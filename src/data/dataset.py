from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for Deepfake Detection.

    Folder Structure:
    datasets/raw/deepfake-dataset/
    ├── real/
    └── fake/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_map = {
            "real": 0,
            "fake": 1
        }

        self.samples = []

        image_extensions = ("*.jpg", "*.jpeg", "*.png")

        for class_name, label in self.class_map.items():
            class_folder = self.root_dir / class_name

            if not class_folder.exists():
                continue

            for extension in image_extensions:
                for image_path in class_folder.rglob(extension):
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
if __name__ == "__main__":

    dataset = DeepfakeDataset(
        root_dir="datasets/raw/deepfake-dataset"
    )

    print(f"Total Images : {len(dataset)}")

    image, label = dataset[0]

    print(f"Image Size : {image.size}")
    print(f"Label      : {label}")
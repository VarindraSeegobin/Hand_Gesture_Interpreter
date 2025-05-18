from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import transforms

transform = transforms.Compose([transforms.Resize((64,6)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class FingersDataset(Dataset):
    def __init__(self, image_directory, labels_directory, transform=transform):
        self.image_directory = image_directory
        self.labels_directory = labels_directory
        self.transform = transform if transform else T.ToTensor()

        self.image_paths = []
        self.label_values = []

        for img_name in os.listdir(image_directory):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(labels_directory, label_name)
            image_path = os.path.join(image_directory, img_name)

            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    try:
                        class_id = int(line.split()[0])
                        if 0 <= class_id <= 5:
                            self.image_paths.append(image_path)
                            self.label_values.append(class_id)
                        else:
                            print(f"Skipping invalid label {class_id} in file: {label_path}")
                    except Exception as e:
                        print(f"Skipping malformed label in {label_path}: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.label_values[idx])

        if self.transform:
            image = self.transform(image)

        return image, label




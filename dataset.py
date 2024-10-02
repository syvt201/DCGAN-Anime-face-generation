import os
from torchvision.io import read_image
import os
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [path for path in os.listdir(data_dir) if path.endswith('.png') or path.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.image_paths[idx])
        img = read_image(path).float()
        if self.transform:
            img = self.transform(img)
        return img
    
class HugggingFaceDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset['train']
        self.transform = transform

    def __len__(self):
        return len(self.dataset)  # Change 'train' to 'test' or 'validation' as needed

    def __getitem__(self, idx):
        # Load the image
        image = self.dataset[idx]['image']  # Access the image path
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image
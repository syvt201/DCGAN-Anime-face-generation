import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as dset
from dataset import ImageDataset, HugggingFaceDataset
import torch.nn as nn
import torch.optim as optim
from DCGan import Generator, Discriminator
from training import training
from datasets import load_dataset

data_dir = "huggingface_dataset"
batch_size = 128
latent_size = 100
fearute_size = 64


# dataset = load_dataset("DrishtiSharma/Anime-Face-Dataset", split = 'train')

# print("Load dataset successfully")
# # Split into smaller dataset
# dataset_split = dataset.train_test_split(test_size=0.3, shuffle=True)

# anime_face_dataset = HugggingFaceDataset(dataset_split, transform=transforms.Compose([
#                                transforms.Resize(64),
#                                transforms.CenterCrop(64),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))

dataset = dset.ImageFolder(root=data_dir,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netD = Discriminator(fearute_size, 3).to(device)
netG = Generator(latent_size, fearute_size, 3).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

optim_d = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_g = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

training(netG, netD, data_loader, batch_size, latent_size, optim_g, optim_d, num_epochs=300)
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.io import read_image, ImageReadMode
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import random
from dataset import ImageDataset
from tqdm import tqdm  
import os
import json
import time
import torchvision.utils as vutils

def train_discriminator(netG, netD, real_images, batch_size, latent_size, criterion, optim_d, device):
    netD.zero_grad()
            
    # Pass real images through discriminator
    output = netD(real_images).view(-1)
    real_label = torch.full(output.shape, 1.0).to(device)
    real_loss = criterion(output, real_label)
    D_x_score = output.mean().item()
    
    # Generate fake images
    noises = torch.randn(batch_size, latent_size, 1, 1).to(device)
    fake_images = netG(noises)
    
    # Pass fake images through discriminator
    output = netD(fake_images).view(-1)
    fake_label = torch.full(output.shape, 0.0).to(device)
    fake_loss = criterion(output, fake_label)
    D_G_score = output.mean().item()
    # Update discriminator weights
    loss_d = real_loss + fake_loss
    
    loss_d.backward()
    optim_d.step()
    
    return netD, loss_d.item(), D_x_score, D_G_score

def train_generator(netG, netD, batch_size, latent_size, criterion, optim_g, device):
    optim_g.zero_grad()
    noises = torch.randn(batch_size, latent_size, 1, 1).to(device)
    fake_images = netG(noises)
    
    D_out_fake = netD(fake_images).view(-1)
    real_label = torch.full(D_out_fake.shape, 1.0).to(device)
    loss_g = criterion(D_out_fake, real_label)
    D_G_score = D_out_fake.mean().item()
    
    loss_g.backward()
    optim_g.step()
    
    return netG, loss_g.item(), D_G_score

def denorm2(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means
 
def training(netG, netD, data_loader, batch_size, latent_size, optim_g, optim_d, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)
    
    netG = netG.to(device)
    netD = netD.to(device)
    
    # Losses
    losses_g = []
    losses_d = []
    history = {}
    
    iter_losses_g = []
    iter_losses_d = []
    epoch_losses_g = []
    epoch_losses_d = []
    
    index = 0
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        start = time.time()
        for i, data in enumerate(data_loader, 0):
            netD.zero_grad()
            # Format batch
            real_images = data[0].to(device)
            # Pass real images through discriminator
            D_real_output = netD(real_images).view(-1)
            b_size = D_real_output.size(0)
            
            # Create real label
            real_label = torch.full((b_size, ), 1.0, dtype=torch.float, device=device)
            
            # Calculate loss on all-real batch
            D_real_loss = criterion(D_real_output, real_label)
            
            # Calculate gradients for D in backward pass
            D_real_loss.backward()
            D_x = D_real_output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_size, 1, 1, device=device)
            
            # Generate fake image batch with G
            fake_images = netG(noise)
            fake_label = torch.full((b_size, ), 0.0, dtype=torch.float, device=device)
            
            # Classify all fake batch with D
            D_fake_output = netD(fake_images.detach()).view(-1)
            
            # Calculate D's loss on the all-fake batch
            D_fake_loss = criterion(D_fake_output, fake_label)
            
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            D_fake_loss.backward()
            D_G_z1 = D_fake_output.mean().item()
            
            # Compute error of D as sum over the fake and the real batches
            D_loss = D_real_loss + D_fake_loss
            
            # Update D
            optim_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            real_label = torch.full((b_size, ), 1.0, dtype=torch.float, device=device)
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            D_fake_output = netD(fake_images).view(-1)
            
            # Calculate G's loss based on this output
            G_loss = criterion(D_fake_output, real_label)
            
            # Calculate gradients for G
            G_loss.backward()
            D_G_z2 = D_fake_output.mean().item()
            
            # Update G
            optim_g.step()
            
            loss_g, loss_d = G_loss.item(), D_loss.item()
            
            if i % 50 == 0:
                print('[%3d/%3d][%3d/%3d]    Loss_D: %.4f    Loss_G: %.4f    D(x): %.4f    D(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(data_loader), loss_g, loss_d, D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            
            iter_losses_g.append(loss_g)
            iter_losses_d.append(loss_d)
            
        # Save generated image every epochs
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        denorm_images = denorm2(fake, *stats)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        index += 1
        save_image(denorm_images, os.path.join('process_300eps', fake_fname), nrow=8)
        print('Saving', fake_fname)
        
        epoch_time = time.time() - start
        
        losses_g_avg = sum(losses_g) / len(losses_g)
        losses_d_avg = sum(losses_d) / len(losses_d)
        
        epoch_losses_d.append(losses_d_avg)
        epoch_losses_g.append(losses_g_avg)
            
        losses_g.clear()
        losses_d.clear()
        
        print('Epoch:[%3d/%3d]    Loss_G: %.4f    Loss_D: %.4f     Time: %.4f' % (epoch, num_epochs, losses_g_avg, losses_d_avg, epoch_time))
    
        if epoch % 5 == 0 or epoch == num_epochs-1:
            torch.save(netD.state_dict(), f'trained_models_300eps/D/Discriminator_{epoch}.pth')
            torch.save(netG.state_dict(), f'trained_models_300eps/G/Generator_{epoch}.pth')
            
    history_path = 'history_300eps.json' 
    
    history = {
        "iter_losses_d": iter_losses_d,
        "iter_losses_g": iter_losses_g,
        "epoch_losses_d": epoch_losses_d,
        "epoch_losses_g": epoch_losses_g
    } 
    with open(history_path, 'w') as json_file:
        json.dump(history, json_file, indent=4)  # `indent` for pretty formatting

    print(f'Dictionary saved to {history_path}')
    
        
    
    
                
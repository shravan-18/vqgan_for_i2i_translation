import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from vqvae import VQVAE
from lpips import LPIPS
from discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from mnist_dataset import MnistDataset
from torch.optim import Adam
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    # Create the dataset
    im_dataset_cls = {
        'underwater': MnistDataset,
    }.get(dataset_config['name'])
    
    train_dataset = im_dataset_cls(split='train',
                                root_path=dataset_config['root_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    train_loader = DataLoader(train_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    
    val_dataset = im_dataset_cls(split='val',
                                root_path=dataset_config['root_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    val_loader = DataLoader(val_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    
    for epoch_idx in range(num_epochs):
        # Training phase
        model.train()
        discriminator.train()
        
        train_recon_losses = []
        train_codebook_losses = []
        train_perceptual_losses = []
        train_disc_losses = []
        train_gen_losses = []
        train_losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im, gt in tqdm(train_loader, desc=f"Training Epoch {epoch_idx+1}"):
            step_count += 1
            im = im.float().to(device)
            gt = gt.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output
            
            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, gt) 
            train_recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            train_codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                train_gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            
            lpips_loss = torch.mean(lpips_model(output, gt))
            train_perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps
            train_losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(gt)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                train_disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        
        # Final optimizer steps for training
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        # Validation phase
        model.eval()
        discriminator.eval()
        
        val_recon_losses = []
        val_codebook_losses = []
        val_perceptual_losses = []
        val_disc_losses = []
        val_gen_losses = []
        val_losses = []
        
        with torch.no_grad():
            for im, gt in tqdm(val_loader, desc=f"Validation Epoch {epoch_idx+1}"):
                im = im.float().to(device)
                gt = gt.float().to(device)
                
                # Fetch autoencoders output(reconstructions)
                model_output = model(im)
                output, z, quantize_losses = model_output
                
                # L2 Loss
                recon_loss = recon_criterion(output, gt)
                val_recon_losses.append(recon_loss.item())
                
                g_loss = (recon_loss +
                          (train_config['codebook_weight'] * quantize_losses['codebook_loss']) +
                          (train_config['commitment_beta'] * quantize_losses['commitment_loss']))
                val_codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
                
                # Adversarial loss only if disc_step_start steps passed
                if step_count > disc_step_start:
                    disc_fake_pred = discriminator(model_output[0])
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.ones(disc_fake_pred.shape,
                                                               device=disc_fake_pred.device))
                    val_gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                    g_loss += train_config['disc_weight'] * disc_fake_loss
                    
                    # Discriminator validation loss
                    fake = output
                    disc_fake_pred = discriminator(fake)
                    disc_real_pred = discriminator(gt)
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.zeros(disc_fake_pred.shape,
                                                                device=disc_fake_pred.device))
                    disc_real_loss = disc_criterion(disc_real_pred,
                                                    torch.ones(disc_real_pred.shape,
                                                               device=disc_real_pred.device))
                    disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                    val_disc_losses.append(disc_loss.item())
                
                lpips_loss = torch.mean(lpips_model(output, gt))
                val_perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
                g_loss += train_config['perceptual_weight'] * lpips_loss
                val_losses.append(g_loss.item())
        
        # Print training and validation metrics
        if len(train_disc_losses) > 0 and len(val_disc_losses) > 0:
            print(
                'Finished epoch: {} | Train - Recon: {:.4f}, Perceptual: {:.4f}, '
                'Codebook: {:.4f}, G Loss: {:.4f}, D Loss: {:.4f} | '
                'Val - Recon: {:.4f}, Perceptual: {:.4f}, '
                'Codebook: {:.4f}, G Loss: {:.4f}, D Loss: {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(train_recon_losses), np.mean(train_perceptual_losses),
                       np.mean(train_codebook_losses), np.mean(train_gen_losses), np.mean(train_disc_losses),
                       np.mean(val_recon_losses), np.mean(val_perceptual_losses),
                       np.mean(val_codebook_losses), np.mean(val_gen_losses), np.mean(val_disc_losses)))
        elif len(train_disc_losses) > 0:
            print(
                'Finished epoch: {} | Train - Recon: {:.4f}, Perceptual: {:.4f}, '
                'Codebook: {:.4f}, G Loss: {:.4f}, D Loss: {:.4f} | '
                'Val - Recon: {:.4f}, Perceptual: {:.4f}, Codebook: {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(train_recon_losses), np.mean(train_perceptual_losses),
                       np.mean(train_codebook_losses), np.mean(train_gen_losses), np.mean(train_disc_losses),
                       np.mean(val_recon_losses), np.mean(val_perceptual_losses), np.mean(val_codebook_losses)))
        else:
            print(
                'Finished epoch: {} | Train - Recon: {:.4f}, Perceptual: {:.4f}, Codebook: {:.4f} | '
                'Val - Recon: {:.4f}, Perceptual: {:.4f}, Codebook: {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(train_recon_losses), np.mean(train_perceptual_losses), np.mean(train_codebook_losses),
                       np.mean(val_recon_losses), np.mean(val_perceptual_losses), np.mean(val_codebook_losses)))
        
        # Save model checkpoints
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['vqvae_discriminator_ckpt_name']))
    
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='underwater.yaml', type=str)
    args = parser.parse_args()
    train(args)

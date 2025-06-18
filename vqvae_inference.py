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
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import piq  # For BRISQUE - install with: pip install piq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    # Convert tensors to numpy arrays and ensure they're in [0, 1] range
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    
    # Calculate PSNR for each image in the batch
    psnr_values = []
    for i in range(img1_np.shape[0]):
        # Convert from CHW to HWC and ensure values are in [0, 1]
        if img1_np.shape[1] == 1:  # Grayscale
            img1_single = img1_np[i, 0, :, :]
            img2_single = img2_np[i, 0, :, :]
        else:  # Color
            img1_single = np.transpose(img1_np[i], (1, 2, 0))
            img2_single = np.transpose(img2_np[i], (1, 2, 0))
        
        psnr_val = psnr(img1_single, img2_single, data_range=1.0)
        psnr_values.append(psnr_val)
    
    return np.array(psnr_values)

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index between two images"""
    # Convert tensors to numpy arrays
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    
    ssim_values = []
    for i in range(img1_np.shape[0]):
        if img1_np.shape[1] == 1:  # Grayscale
            img1_single = img1_np[i, 0, :, :]
            img2_single = img2_np[i, 0, :, :]
            ssim_val = ssim(img1_single, img2_single, data_range=1.0)
        else:  # Color
            img1_single = np.transpose(img1_np[i], (1, 2, 0))
            img2_single = np.transpose(img2_np[i], (1, 2, 0))
            ssim_val = ssim(img1_single, img2_single, data_range=1.0, channel_axis=2)
        
        ssim_values.append(ssim_val)
    
    return np.array(ssim_values)

def calculate_brisque(img):
    """Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score"""
    # BRISQUE expects images in [0, 1] range
    # Lower BRISQUE scores indicate better image quality
    try:
        brisque_scores = []
        for i in range(img.shape[0]):
            # Extract single image and ensure it's in correct format
            single_img = img[i:i+1]  # Keep batch dimension
            brisque_score = piq.brisque(single_img, data_range=1.0, reduction='none')
            brisque_scores.append(brisque_score.item())
        return np.array(brisque_scores)
    except Exception as e:
        print(f"Warning: BRISQUE calculation failed: {e}")
        return np.array([0.0] * img.shape[0])

def infer(args):
    print("=== VQ-VAE Inference with Metrics Calculation ===")
    print(f"Using device: {device}")
    
    # Read the config file #
    print("Loading configuration...")
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error loading config: {exc}")
            return
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    print(f"Dataset: {dataset_config['name']}")
    print(f"Image size: {dataset_config['im_size']}x{dataset_config['im_size']}")
    print(f"Image channels: {dataset_config['im_channels']}")
    
    # Set the desired seed value #
    seed = train_config['seed']
    print(f"Setting random seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Create the model and dataset #
    print("Initializing VQ-VAE model...")
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    
    # Load trained model weights
    checkpoint_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
    print(f"Loading model weights from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Create validation dataset
    print("Loading validation dataset...")
    im_dataset_cls = {
        'underwater': MnistDataset,
    }.get(dataset_config['name'])
    
    if im_dataset_cls is None:
        print(f"Error: Unknown dataset type: {dataset_config['name']}")
        return
    
    val_dataset = im_dataset_cls(split='val',
                                root_path=dataset_config['root_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    val_loader = DataLoader(val_dataset,
                           batch_size=train_config['autoencoder_batch_size'],
                           shuffle=False)  # No shuffling for consistent evaluation
    
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Initialize LPIPS model
    print("Initializing LPIPS model...")
    lpips_model = LPIPS().eval().to(device)
    
    # Initialize metric storage
    all_psnr_scores = []
    all_ssim_scores = []
    all_lpips_scores = []
    all_brisque_original = []
    all_brisque_reconstructed = []
    
    print("\n=== Starting Inference ===")
    
    with torch.no_grad():
        for batch_idx, (im, gt) in enumerate(tqdm(val_loader, desc="Processing validation images")):
            im = im.float().to(device)
            gt = gt.float().to(device)
            
            # Forward pass through VQ-VAE
            model_output = model(im)
            reconstructed, z, quantize_losses = model_output
            
            # Convert images to [0, 1] range for metric calculations
            # Assuming input images are in [-1, 1] range
            original_01 = (im + 1) / 2
            reconstructed_01 = torch.clamp((reconstructed + 1) / 2, 0, 1)
            gt_01 = (gt + 1) / 2
            
            # Calculate PSNR
            psnr_scores = calculate_psnr(reconstructed_01, gt_01)
            all_psnr_scores.extend(psnr_scores)
            
            # Calculate SSIM
            ssim_scores = calculate_ssim(reconstructed_01, gt_01)
            all_ssim_scores.extend(ssim_scores)
            
            # Calculate LPIPS
            lpips_scores = lpips_model(reconstructed, gt)  # LPIPS expects [-1, 1] range
            all_lpips_scores.extend(lpips_scores.cpu().numpy())
            
            # Calculate BRISQUE for original and reconstructed images
            brisque_orig = calculate_brisque(original_01)
            brisque_recon = calculate_brisque(reconstructed_01)
            all_brisque_original.extend(brisque_orig)
            all_brisque_reconstructed.extend(brisque_recon)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(val_loader)} batches")
                print(f"  Current batch - PSNR: {np.mean(psnr_scores):.3f}, "
                      f"SSIM: {np.mean(ssim_scores):.3f}, "
                      f"LPIPS: {np.mean(lpips_scores.cpu().numpy()):.3f}")
    
    # Convert to numpy arrays
    all_psnr_scores = np.array(all_psnr_scores)
    all_ssim_scores = np.array(all_ssim_scores)
    all_lpips_scores = np.array(all_lpips_scores)
    all_brisque_original = np.array(all_brisque_original)
    all_brisque_reconstructed = np.array(all_brisque_reconstructed)
    
    print(f"\n=== Final Results (Total images: {len(all_psnr_scores)}) ===")
    print(f"PSNR:")
    print(f"  Mean: {np.mean(all_psnr_scores):.4f} ± {np.std(all_psnr_scores):.4f}")
    print(f"  Min: {np.min(all_psnr_scores):.4f}, Max: {np.max(all_psnr_scores):.4f}")
    
    print(f"\nSSIM:")
    print(f"  Mean: {np.mean(all_ssim_scores):.4f} ± {np.std(all_ssim_scores):.4f}")
    print(f"  Min: {np.min(all_ssim_scores):.4f}, Max: {np.max(all_ssim_scores):.4f}")
    
    print(f"\nLPIPS:")
    print(f"  Mean: {np.mean(all_lpips_scores):.4f} ± {np.std(all_lpips_scores):.4f}")
    print(f"  Min: {np.min(all_lpips_scores):.4f}, Max: {np.max(all_lpips_scores):.4f}")
    
    print(f"\nBRISQUE (Original Images):")
    print(f"  Mean: {np.mean(all_brisque_original):.4f} ± {np.std(all_brisque_original):.4f}")
    print(f"  Min: {np.min(all_brisque_original):.4f}, Max: {np.max(all_brisque_original):.4f}")
    
    print(f"\nBRISQUE (Reconstructed Images):")
    print(f"  Mean: {np.mean(all_brisque_reconstructed):.4f} ± {np.std(all_brisque_reconstructed):.4f}")
    print(f"  Min: {np.min(all_brisque_reconstructed):.4f}, Max: {np.max(all_brisque_reconstructed):.4f}")
    
    # Save results to file
    results_file = os.path.join(train_config['task_name'], 'inference_metrics.txt')
    print(f"\nSaving detailed results to: {results_file}")
    
    with open(results_file, 'w') as f:
        f.write("VQ-VAE Inference Metrics\n")
        f.write("========================\n\n")
        f.write(f"Total images evaluated: {len(all_psnr_scores)}\n\n")
        
        f.write(f"PSNR: {np.mean(all_psnr_scores):.4f} ± {np.std(all_psnr_scores):.4f}\n")
        f.write(f"SSIM: {np.mean(all_ssim_scores):.4f} ± {np.std(all_ssim_scores):.4f}\n")
        f.write(f"LPIPS: {np.mean(all_lpips_scores):.4f} ± {np.std(all_lpips_scores):.4f}\n")
        f.write(f"BRISQUE (Original): {np.mean(all_brisque_original):.4f} ± {np.std(all_brisque_original):.4f}\n")
        f.write(f"BRISQUE (Reconstructed): {np.mean(all_brisque_reconstructed):.4f} ± {np.std(all_brisque_reconstructed):.4f}\n")
    
    print("Inference completed successfully!")
    
    return {
        'psnr': all_psnr_scores,
        'ssim': all_ssim_scores,
        'lpips': all_lpips_scores,
        'brisque_original': all_brisque_original,
        'brisque_reconstructed': all_brisque_reconstructed
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VQ-VAE inference with metrics')
    parser.add_argument('--config', dest='config_path',
                        default='underwater.yaml', type=str,
                        help='Path to the configuration file')
    args = parser.parse_args()
    infer(args)
    
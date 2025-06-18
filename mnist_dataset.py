import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    
    def __init__(self, split, root_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        # Load all images first
        all_input_imgs, all_gt_imgs = self.load_all_images(root_path)
        
        # Split into train/val (80-20 ratio)
        self.input_imgs, self.gt_imgs = self.split_images(all_input_imgs, all_gt_imgs, split)
        
        # # Whether to load images and call vae or to load latents
        # if use_latents and latent_path is not None:
        #     latent_maps = load_latents(latent_path)
        #     if len(latent_maps) == len(self.images):
        #         self.use_latents = True
        #         self.latent_maps = latent_maps
        #         print('Found {} latents'.format(len(self.latent_maps)))
        #     else:
        #         print('Latents not found')

    def load_all_images(self, root_path):
        r"""
        Gets all images from the root_path path specified
        and stacks them all up
        :param root_path: root path containing degraded and reference folders
        :return: lists of input and gt image paths
        """
        assert os.path.exists(root_path), "images path {} does not exist".format(root_path)
        assert len(os.listdir(root_path)) == 2, "paired images not found"
        
        input_imgs = []
        gt_imgs = []
        input_imgs_dir = os.path.join(root_path, 'degraded')
        gt_imgs_dir = os.path.join(root_path, 'reference')

        # print(f"input_imgs_dir: {input_imgs_dir}")
        # print(f"gt_imgs_dir: {gt_imgs_dir}")

        # print(f"len of input_imgs_dir: {len(os.listdir(input_imgs_dir))}")
        # print(f"len of gt_imgs_dir: {len(os.listdir(gt_imgs_dir))}")
        
        for img_fn in os.listdir(input_imgs_dir):
            input_imgs.append(os.path.join(input_imgs_dir, img_fn))
        
        for img_fn in os.listdir(gt_imgs_dir):
            gt_imgs.append(os.path.join(gt_imgs_dir, img_fn))
        
        assert len(input_imgs) == len(gt_imgs), "images not equal in input and GT dirs"
        print(f"Length of input imgs list: {len(input_imgs)}")
        print(f"Length of input GT list: {len(gt_imgs)}")
        
        # Sort both lists to ensure consistent pairing
        input_imgs.sort()
        gt_imgs.sort()
        
        return input_imgs, gt_imgs

    def split_images(self, input_imgs, gt_imgs, split):
        r"""
        Split the image lists into train/val based on 80-20 ratio
        :param input_imgs: list of input image paths
        :param gt_imgs: list of gt image paths  
        :param split: 'train' or 'val'
        :return: split input and gt image lists
        """
        import random
        
        # Create paired list and shuffle with fixed seed for reproducibility
        paired_imgs = list(zip(input_imgs, gt_imgs))
        random.seed(42)  # Fixed seed for reproducible splits
        random.shuffle(paired_imgs)
        
        # Calculate split indices
        total_imgs = len(paired_imgs)
        train_size = int(0.8 * total_imgs)
        
        if split == 'train':
            split_paired = paired_imgs[:train_size]
        elif split == 'val':
            split_paired = paired_imgs[train_size:]
        else:
            raise ValueError("Split must be 'train' or 'val'")
        
        # Unzip back to separate lists
        if len(split_paired) > 0:
            split_input, split_gt = zip(*split_paired)
            return list(split_input), list(split_gt)
        else:
            return [], []
        
    def load_images(self, root_path):
        r"""
        Legacy method - kept for backward compatibility
        Gets all images from the root_path path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        return self.load_all_images(root_path)
    
    def __len__(self):
        return len(self.input_imgs)
    
    def __getitem__(self, index):
        # input image
        im_input = Image.open(self.input_imgs[index])
        # Resize input image to consistent dimensions
        im_input = torchvision.transforms.Resize((self.im_size, self.im_size))(im_input)
        im_input_tensor = torchvision.transforms.ToTensor()(im_input)
        im_input_tensor = (2 * im_input_tensor) - 1 # Convert to -1 to 1 range.
        
        # GT image
        im_gt = Image.open(self.gt_imgs[index])
        # Resize GT image to consistent dimensions
        im_gt = torchvision.transforms.Resize((self.im_size, self.im_size))(im_gt)
        im_gt_tensor = torchvision.transforms.ToTensor()(im_gt)
        im_gt_tensor = (2 * im_gt_tensor) - 1 # Convert to -1 to 1 range.
        
        return im_input_tensor, im_gt_tensor
    
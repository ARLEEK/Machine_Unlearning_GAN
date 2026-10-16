import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import os
import argparse
import glob
from PIL import Image
import numpy as np
import joblib

# Monkey patch Adam to ensure compatibility with older GAN Zoo code
# older versions of the zoo code pass betas as a list, simpler torch versions expect tuple
_old_adam = optim.Adam
def _new_adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, **kwargs):
    if not isinstance(betas, tuple):
        betas = tuple(betas)
    betas = (float(betas[0]), float(betas[1]))
    return _old_adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)
optim.Adam = _new_adam

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    z_dim = 512
    resolution = 256 # PGAN native resolution for this checkpoint
    batch_size = 8 # Default batch size
    lr = 0.001
    checkpoint_dir = 'checkpoints_pgan'
    sample_dir = 'samples_pgan'
    data_path = './data/celeba_hq'
    model_repo = 'facebookresearch/pytorch_GAN_zoo:hub'
    model_name = 'celebAHQ-256'

# SVDD Encoder Definition
class Encoder(nn.Module):
    class AddGaussianNoise(nn.Module):
        def __init__(self, mean=0.0, std=0.05):
            super().__init__()
            self.mean = mean
            self.std = std

        def forward(self, x):
            if not self.training:
                return x
            return x + torch.randn_like(x) * self.std + self.mean

    def __init__(self):
        super().__init__()
        # Regularization
        self.noise = self.AddGaussianNoise(0.0, 0.15)

        self.feature_extractor = nn.Sequential(
            # 1 -> 3 channels for CelebA
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 128 channels come out of the last conv layer
        self.fc = nn.Linear(128, 32)

    def forward(self, x):
        x = self.noise(x)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_svdd_components(device):
    """
    Loads the SVDD model, center, and threshold.
    """
    print("Loading SVDD Classifier components...")
    
    # 1. Load Model
    svdd = Encoder().to(device)
    try:
        # Try state dict first
        state_dict = torch.load('gan/celeb_svdd.pth', map_location=device)
        svdd.load_state_dict(state_dict)
        print("Loaded SVDD weights from gan/celeb_svdd.pth")
    except Exception as e:
        print(f"Could not load state dict from pth: {e}. Trying joblib...")
        try:
            loaded_model = joblib.load('gan/model.joblib')
            # Extract state dict from loaded object if possible, or use it directly
            if isinstance(loaded_model, nn.Module):
                svdd = loaded_model.to(device)
            else:
                print("Joblib loaded object is not a module.")
        except Exception as e2:
            print(f"Failed to load SVDD model: {e2}")
            return None, None, None

    svdd.eval()

    # 2. Load Center and Threshold
    try:
        center = torch.from_numpy(np.load('gan/center.npy')).to(device)
        threshold = float(np.load('gan/threshold.npy'))
        print(f"Loaded center (shape {center.shape}) and threshold ({threshold})")
    except Exception as e:
        print(f"Failed to load center/threshold: {e}")
        return None, None, None
        
    return svdd, center, threshold


# 2. Dataset
class CelebADataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_files = glob.glob(os.path.join(root_dir, '*.jpg')) + \
                           glob.glob(os.path.join(root_dir, '*.png'))
        if not self.image_files:
            print(f"Warning: No images found in {root_dir}")
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.randn(3, 256, 256) # Fallback


# 3. Model Loading & Wrappers
def load_pretrained_pgan(pretrained=True):
    """
    Loads the Progressive GAN trained on CelebA-HQ from PyTorch Hub.
    """
    print(f"Loading PGAN ({Config.model_name}) from {Config.model_repo}...")
    use_gpu = (Config.device == 'cuda')
    
    try:
        model = torch.hub.load(
            Config.model_repo, 
            'PGAN', 
            model_name=Config.model_name, 
            pretrained=pretrained, 
            useGPU=use_gpu
        )
        return model
    except Exception as e:
        print(f"Failed to load PGAN: {e}")
        import traceback
        traceback.print_exc()
        raise e


# 4. Training Loop (Fine-Tuning / Retraining)
def gradient_penalty(netD, real_data, fake_data, device='cpu'):
    # Simple WGAN-GP gradient penalty implementation
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_(True)
    
    disc_interpolates = netD(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10 
    return gradient_penalty

def train(args):
    print("Initializing Training...")
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.sample_dir, exist_ok=True)
    
    # Data
    dataset = CelebADataset(args.data_path, image_size=Config.resolution)
    if len(dataset) == 0:
        print("Error: Dataset is empty. Please provide valid image path.")
        return
        
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    
    # Model
    # load the pretrained model
    hub_model = load_pretrained_pgan(pretrained=True)
    
    netG = hub_model.netG.to(Config.device)
    netD = hub_model.netD.to(Config.device)
    
    # Optimizers
    # We create fresh optimizers for stability in fine-tuning, or try to load state if accessible
    # Hub model stores optimizers, but might be easier to just re-init for fine-tuning
    optimizerG = optim.Adam(netG.parameters(), lr=Config.lr, betas=(0.0, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=Config.lr, betas=(0.0, 0.99))
    
    start_step = 0
    
    print(f"Starting fine-tuning on {args.data_path}...")
    
    for epoch in range(args.epochs):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(Config.device)
            current_batch_size = real_imgs.size(0)
            
            
            # Train Discriminator
            optimizerD.zero_grad()
            
            # Real
            pred_real = netD(real_imgs)
            loss_d_real = -pred_real.mean()
            
            # Fake
            z = torch.randn(current_batch_size, Config.z_dim, device=Config.device)
            fake_imgs = netG(z)
            pred_fake = netD(fake_imgs.detach())
            loss_d_fake = pred_fake.mean()
            
            # GP
            gp = gradient_penalty(netD, real_imgs, fake_imgs.detach(), Config.device)
            
            # Total D Loss (WGAN-GP)
            loss_d = loss_d_real + loss_d_fake + gp
            loss_d.backward()
            optimizerD.step()
            
            # Train Generator
            if i % 1 == 0: # n_critic=1 for now, standard is 5 usually
                optimizerG.zero_grad()
                
                # Regenerate for G update to keep graph clean
                # fake_imgs = netG(z)
                pred_fake_g = netD(fake_imgs)
                
                loss_g = -pred_fake_g.mean()
                loss_g.backward()
                optimizerG.step()
                
                step_num = epoch * len(dataloader) + i + start_step
                
                if i % 50 == 0:
                     print(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(dataloader)}] "
                           f"D Loss: {loss_d.item():.4f} G Loss: {loss_g.item():.4f}")

                if step_num % args.save_interval == 0 and step_num > 0:
                    # Save Checkpoint
                    ckpt_path = os.path.join(Config.checkpoint_dir, f'pgan_finetune_{step_num}.pt')
                    torch.save({
                        'netG': netG.state_dict(),
                        'netD': netD.state_dict(),
                        'optimizerG': optimizerG.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        'step': step_num
                    }, ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")
                    
                    # Save Sample
                    with torch.no_grad():
                        sample_imgs = (fake_imgs[:16] + 1) / 2.0
                        utils.save_image(sample_imgs, os.path.join(Config.sample_dir, f'train_step_{step_num}.png'), nrow=4)

    # Save final model
    torch.save({
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
    }, os.path.join(Config.checkpoint_dir, 'pgan_final.pt'))
    print("Training Complete.")


# 5. Generation
def generate(args):
    print("Loading Model for Generation...")
    
    # If a custom checkpoint is provided, load it. Otherwise use the Hub pretrained model.
    if args.checkpoint:
        # We need the architecture to load weights into. 
        # We can use the hub model as a shell.
        hub_model = load_pretrained_pgan(pretrained=False)
        netG = hub_model.netG.to(Config.device)
        
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=Config.device)
        if 'netG' in checkpoint:
            netG.load_state_dict(checkpoint['netG'])
        else:
            # Maybe it's just the state dict directly?
            netG.load_state_dict(checkpoint)
    else:
        # Use the pretrained Hub model directly
        print("Using pretrained model from PyTorch Hub")
        hub_model = load_pretrained_pgan(pretrained=True)
        netG = hub_model.netG.to(Config.device)
        
    netG.eval()
    
    print(f"Generating {args.num_samples} images...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    batch_size = Config.batch_size
    n_batches = (args.num_samples + batch_size - 1) // batch_size
    
    count = 0
    with torch.no_grad():
        for i in range(n_batches):
            current_batch = min(batch_size, args.num_samples - count)
            z = torch.randn(current_batch, Config.z_dim, device=Config.device)
            
            imgs = netG(z)
            
            # Process images (-1,1 -> 0,1)
            imgs = (imgs + 1.0) / 2.0
            imgs = torch.clamp(imgs, 0, 1)
            
            for k in range(imgs.size(0)):
                utils.save_image(imgs[k], os.path.join(args.output_dir, f'gen_{count:04d}.png'))
                count += 1
                
    print(f"Done. Saved to {args.output_dir}")

def unlearn(args):
    print("Initializing Unlearning/Self-Correction Loop...")
    os.makedirs(Config.sample_dir, exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # Load SVDD
    svdd, center, threshold = load_svdd_components(Config.device)
    if svdd is None:
        print("Cannot proceed without SVDD classifier.")
        return

    # Load PGAN (Trainable)
    hub_model = load_pretrained_pgan(pretrained=True)
    netG = hub_model.netG.to(Config.device)
    netD = hub_model.netD.to(Config.device)
    
    # Load PGAN (Frozen Reference for Anchor)
    hub_model_ref = load_pretrained_pgan(pretrained=True)
    netG_ref = hub_model_ref.netG.to(Config.device)
    netG_ref.eval()
    for p in netG_ref.parameters():
        p.requires_grad = False
    
    # Optimizers - Lower LR for stability (controllable via args)
    # Using a lower LR helps prevent catastrophic forgetting/noise
    print(f"Setting learning rate to {args.lr} for stability.")
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.0, 0.99))
    
    # Transformation for SVDD (256 -> 64)
    # PGAN outputs are already normalized in range [-1, 1], which corresponds to the 
    # Normalize((0.5,), (0.5,)) used in SVDD training if input was [0,1].
    resize_transform = transforms.Resize((64, 64))
    
    print(f"Starting unlearning for {args.epochs} epochs...")
    print(f"Batch Size: {args.batch_size}, n_critic: {args.n_critic}, Anchor Weight: {args.anchor_weight}")
    
    for epoch in range(args.epochs):
        
        # Generation Step
        # Generate a batch
        current_batch_size = args.batch_size
        z = torch.randn(current_batch_size, Config.z_dim, device=Config.device)
        
        # Generate images
        netG.eval() # Use eval for generation to avoid batchnorm shifts during filtering? 
                    # Actually we want gradients later, so maybe train? 
                    # But for filtering we just want the current state.
        
        with torch.no_grad():
            fake_imgs_raw = netG(z)
            
            # SVDD Classification
            # Resize
            svdd_input = resize_transform(fake_imgs_raw)
            
            # Forward
            features = svdd(svdd_input)
            dist = torch.sum((features - center)**2, dim=1)
            
            # Decision:
            # Notebook: "Normal" (Glasses) if -dist > best_thresh  => dist < -best_thresh
            # Our threshold loaded is best_thresh (negative value).
            # So dist < -threshold implies Glasses.
            # We want to UNLEARN glasses. So we want images where dist >= -threshold.
            # i.e. NOT Glasses.
            
            # threshold is negative, e.g. -1.9e-6. -threshold is positive.
            limit = -threshold
            
            has_glasses = dist < limit
            
            # Filter
            # indices of images WITHOUT glasses
            keep_mask = ~has_glasses
            filtered_imgs = fake_imgs_raw[keep_mask]
            
            n_glasses = has_glasses.sum().item()
            prob = n_glasses / current_batch_size
            
            print(f"Epoch {epoch}: Generated {current_batch_size} images. "
                  f"With Glasses: {n_glasses} ({prob:.2%}). "
                  f"Kept (Clean): {len(filtered_imgs)}")
            
            # Save sample
            if epoch % 10 == 0:
                utils.save_image(
                    (fake_imgs_raw[:8] + 1) / 2.0, 
                    os.path.join(Config.sample_dir, f'unlearn_epoch_{epoch}.png'), 
                    nrow=int(np.ceil(np.sqrt(current_batch_size)))
                )
                
            # Print sample to console (scaled down for ASCII/Base64 if we had a viewer, but for now just path)
            # Or we could try to implement a simple ASCII visualizer or just rely on the file save.
            # User wants to "print the images being generated". 
            # We can't easily print high-res images to text console, but we can print the path.
            print(f"Saved sample to {os.path.join(Config.sample_dir, f'unlearn_epoch_{epoch}.png')}")

        
        # Retraining Step
        if len(filtered_imgs) > 1: # Need at least something to train on
            netG.train()
            netD.train()
            
            # We treat 'filtered_imgs' as Real data
            real_data = filtered_imgs.detach()
            
            # 1. Train Discriminator (n_critic times)
            mean_d_loss = 0
            for _ in range(args.n_critic):
                optimizerD.zero_grad()
                
                # "Real" (Filtered Fakes)
                # Note: reusing the same batch of filtered images for all critic steps
                # Ideally we'd have a buffer, but this works for "fine-tuning" on this batch
                pred_real = netD(real_data)
                loss_d_real = -pred_real.mean()
                
                # "Fake" (New generation from G)
                z_new = torch.randn(len(real_data), Config.z_dim, device=Config.device)
                fake_imgs_new = netG(z_new)
                pred_fake = netD(fake_imgs_new.detach())
                loss_d_fake = pred_fake.mean()
                
                # GP
                gp = gradient_penalty(netD, real_data, fake_imgs_new.detach(), Config.device)
                
                loss_d = loss_d_real + loss_d_fake + gp
                loss_d.backward()
                optimizerD.step()
                mean_d_loss += loss_d.item()
            
            mean_d_loss /= args.n_critic
            
            # 2. Train Generator
            optimizerG.zero_grad()
            
            # Re-generate to get graph
            z_final = torch.randn(len(real_data), Config.z_dim, device=Config.device)
            fake_imgs_final = netG(z_final)
            
            # GAN Loss
            pred_fake_g = netD(fake_imgs_final)
            loss_g_gan = -pred_fake_g.mean()
            
            # Anchor Loss (L1 distance to original model outputs)
            # Helps prevent "face transforming too much"
            if args.anchor_weight > 0:
                with torch.no_grad():
                    ref_imgs = netG_ref(z_final)
                loss_anchor = F.l1_loss(fake_imgs_final, ref_imgs)
            else:
                loss_anchor = torch.tensor(0.0, device=Config.device)
                
            loss_g = loss_g_gan + (args.anchor_weight * loss_anchor)
            
            loss_g.backward()
            optimizerG.step()
            
            if epoch % 1 == 0: # Print every step for feedback
                print(f"  [Train] D Loss: {mean_d_loss:.4f} G Loss: {loss_g.item():.4f} (Anchor: {loss_anchor.item():.4f})")
        else:
            print("  [Train] Skipped update (not enough clean images)")
            
    # Save unlearned model
    torch.save({
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
    }, os.path.join(Config.checkpoint_dir, 'pgan_unlearned.pt'))
    print("Unlearning Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive GAN CelebA-HQ (Simulated via Hub & Wrapper)")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train Parser
    train_parser = subparsers.add_parser('train', help='Train (Fine-tune) the model')
    train_parser.add_argument('--data_path', type=str, default=Config.data_path, help='Path to images')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--save_interval', type=int, default=500, help='Steps between saving checkpoints')
    
    # Generate Parser
    gen_parser = subparsers.add_parser('generate', help='Generate images')
    gen_parser.add_argument('--num_samples', type=int, default=16)
    gen_parser.add_argument('--output_dir', type=str, default='generated_samples_pgan')
    gen_parser.add_argument('--checkpoint', type=str, default=None, help='Path to custom checkpoint. If None, uses official Hub weights.')
    
    # Unlearn Parser
    unlearn_parser = subparsers.add_parser('unlearn', help='Unlearn "Glasses" feature using SVDD')
    unlearn_parser.add_argument('--epochs', type=int, default=100, help='Number of unlearning steps/epochs')
    unlearn_parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation and retraining')
    unlearn_parser.add_argument('--n_critic', type=int, default=5, help='Discriminator steps per Generator step')
    unlearn_parser.add_argument('--anchor_weight', type=float, default=100.0, help='Weight for anchor loss (to preserve identity)')
    unlearn_parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for unlearning (default: 5e-5)')




    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'generate':
        generate(args)
    elif args.command == 'unlearn':
        unlearn(args)

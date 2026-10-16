import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf := ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

# wrapper
class CustomGan:
    def __init__(self, nz=100, nc=3, device="cpu"):
        self.nz = nz
        self.device = device
        self.netG = Generator(nz=nz, nc=nc).to(device)
        self.netD = Discriminator(nc=nc).to(device)

def train_gan(
    dataloader,
    device,
    nz=100,
    n_epochs=25,
    lr=2e-4,
    beta1=0.5,
    out_dir="runs/dcgan",
    nc=3,
    save_every=5,
    sample_size=64,
):
    os.makedirs(out_dir, exist_ok=True)

    gan = CustomGan(nz=nz, nc=nc, device=device)
    netG, netD = gan.netG, gan.netD

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(sample_size, nz, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label, fake_label = 1.0, 0.0

    print("Starting training loop...")
    for epoch in range(1, n_epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        iters = 0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}")):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)
            b_size = imgs.size(0)
            iters += 1

            # --- Update D ---
            netD.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            lossD_real = criterion(netD(imgs), label)
            lossD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            lossD_fake = criterion(netD(fake.detach()), label)
            lossD_fake.backward()
            optimizerD.step()
            lossD = lossD_real + lossD_fake

            # --- Update G ---
            netG.zero_grad()
            label.fill_(real_label)
            lossG = criterion(netD(fake), label)
            lossG.backward()
            optimizerG.step()

            epoch_d_loss += lossD.item()
            epoch_g_loss += lossG.item()

        print(f"Epoch {epoch} | D loss: {epoch_d_loss/iters:.4f} | G loss: {epoch_g_loss/iters:.4f}")

        with torch.no_grad():
            samples = netG(fixed_noise).detach()
            save_image(samples, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"), nrow=8, normalize=True)

        if epoch % save_every == 0 or epoch == n_epochs:
            torch.save({
                "epoch": epoch,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
            }, os.path.join(out_dir, f"dcgan_checkpoint_epoch_{epoch:03d}.pt"))

    torch.save(netG.state_dict(), os.path.join(out_dir, "generator_final.pth"))
    torch.save(netD.state_dict(), os.path.join(out_dir, "discriminator_final.pth"))
    print("Training complete. Models and sample images are saved to:", out_dir)
    return netG, netD

if __name__ == "__main__":
    print("Import this file in run_gan.py for training")
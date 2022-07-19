import torch
from dataset import RainyCleanDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, nm):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)
    acrescimo = 0

    for idx, (clean, rainy) in enumerate(loop):
        clean = clean.to(config.DEVICE)
        rainy = rainy.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_clean = gen_H(clean)
            D_H_real = disc_H(rainy)
            D_H_fake = disc_H(fake_clean.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_rainy = gen_Z(rainy)
            D_Z_real = disc_Z(clean)
            D_Z_fake = disc_Z(fake_rainy.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # Sum of losses
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_clean)
            D_Z_fake = disc_Z(fake_rainy)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_clean = gen_Z(fake_clean)
            cycle_rainy = gen_H(fake_rainy)
            cycle_clean_loss = l1(clean, cycle_clean)
            cycle_rainy_loss = l1(rainy, cycle_rainy)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_clean = gen_Z(clean)
            identity_rainy = gen_H(rainy)
            identity_clean_loss = l1(clean, identity_clean)
            identity_rainy_loss = l1(rainy, identity_rainy)

            # Sum Losses
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_clean_loss * config.LAMBDA_CYCLE
                + cycle_rainy_loss * config.LAMBDA_CYCLE
                + identity_rainy_loss * config.LAMBDA_IDENTITY
                + identity_clean_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_clean*0.5+0.5, f"saved_images/clean_{idx}_{nm}.png")
            save_image(fake_rainy*0.5+0.5, f"saved_images/rainy_{idx}_{nm}.png")
            nm = nm + f"{acrescimo}"
        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))



def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = RainyCleanDataset(
        root_rain=config.TRAIN_DIR + "/clean", root_clean=config.TRAIN_DIR + "/rainy", transform=config.transforms
    )
    val_dataset = RainyCleanDataset(
       root_rain="Images/val/clean", root_clean="Images/val/rainy", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        rodada = 0
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, nm=f"{epoch};{rodada}")

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()
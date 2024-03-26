from model import *
from dataset_ import *
from utils import *

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import argparse
import yaml
from polip import decider

import time

device = decider("mps")

parser = argparse.ArgumentParser(
    description = "Train StackGAN"
)
parser.add_argument("--stage", type = int, default = 1)
parser.add_argument("--yaml_path", type = str, default = "cfg/cfg_stage1.yaml")
parser.add_argument("--data_path", type = str, default = "data/images")
parser.add_argument("--embedding_path", type = str, default = "data/embedding/embeddings.pkl")
args = parser.parse_args()


## yaml loading

with open(args.yaml_path, "r") as file:
    config = yaml.safe_load(file)
## model params
gf_dim = int(config["model_params"]["gf_dim"])
z_dim = int(config["model_params"]["z_dim"])
text_dim = int(config["model_params"]["text_dim"])
df_dim = int(config["model_params"]["df_dim"])
condition_dim = int(config["model_params"]["condition_dim"])

## training params
g_lr = float(config["training_params"]["g_lr"])
d_lr = float(config["training_params"]["d_lr"])
batch_size = int(config["training_params"]["batch_size"])
kl_coeff = float(config["training_params"]["kl_coeff"])
snapshot_interval = int(config["training_params"]["snapshot_interval"])
model_dir = str(config["training_params"]["model_dir"])
image_dir = str(config["training_params"]["image_dir"])
max_epoch = int(config["training_params"]["max_epoch"])
lr_decay_step = int(config["training_params"]["lr_decay_step"])
r_num = int(config["training_params"]["r_num"])

#image params
image_size = int(config["images_params"]["image_size"])

def train(data_loader, stage=1, STAGE1_G=STAGE1_G, STAGE1_D=STAGE1_D, STAGE2_D=STAGE2_D, STAGE2_G=STAGE2_G,
          ):
    if stage == 1:
        netG, netD = load_nns_stage1(STAGE1_G, STAGE1_D, gf_dim,
                                     df_dim, condition_dim, z_dim, text_dim, device)
    else:
        netG, netD = load_nns_stage2(STAGE1_G, STAGE2_G, STAGE2_D,
                                     gf_dim, condition_dim, z_dim, r_num, df_dim, text_dim, device)
    g_lr = 0.0002
    d_lr = 0.0002
    nz = z_dim
    noise = torch.randn((batch_size, nz), dtype=torch.float32).to(device)
    fixed_noise = torch.randn((batch_size, nz), dtype=torch.float32).normal_(0, 1).to(device)
    real_labels = torch.ones((batch_size,), dtype=torch.float32).to(device)
    fake_labels = torch.zeros((batch_size,), dtype=torch.float32).to(device)
    optimizer_d = torch.optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    netg_para = []
    for p in netG.parameters():
        if p.requires_grad:
            netg_para.append(p)
    optimizer_g = torch.optim.Adam(netg_para, lr=g_lr, betas=(0.5, 0.999))
    count = 0
    
    d_losses = []
    g_losses = []
    kl_losses = []
    
    for epoch in range(max_epoch):
        start = time.time()
        
        if epoch % lr_decay_step == 0 and epoch > 0:
            g_lr *= 0.5
            for param_group in optimizer_g.param_groups:
                param_group["lr"] = d_lr
            d_lr *= 0.5
            for param_group in optimizer_d.param_groups:
                param_group["lr"] = d_lr
        for i, data in enumerate(data_loader, 0):
            real_img_cpu, txt_embedding, image_path = data
            real_img_cpu, txt_embedding = real_img_cpu.to(device), txt_embedding.to(device)
            noise.data.normal_(0, 1)
            if txt_embedding.size(0) != noise.size(0):
                print(f"Skipping iteration {i} due to tensor size mismatch")
                continue
            _, fake_imgs, mu, logvar = netG(txt_embedding, noise)
            netD.zero_grad()
            errd, errd_real, errd_wrong, errd_fake = compute_discriminator_loss(netD, real_img_cpu, fake_imgs, real_labels, fake_labels, mu)
            errd.backward()
            d_losses.append(errd.item())
            optimizer_d.step()
            netG.zero_grad()
            errg = compute_generator_loss(netD, fake_imgs, real_labels, mu)
            kl_loss = KL_loss(mu, logvar)
            kl_losses.append(kl_loss.item())
            errg_total = errg + kl_loss * kl_coeff
            g_losses.append(errg_total.item())
            errg_total.backward()
            optimizer_g.step()
            count += 1
            
            # clear_output(wait = True)
            # plt.figure(figsize=(10, 5))
            # plt.title(f"Training Losses after Epoch {epoch}")
            # plt.plot(d_losses, label="Discriminator Loss")
            # plt.plot(g_losses, label="Generator Loss")
            # plt.plot(kl_losses, label="KL Loss")
            # plt.xlabel("Iterations")
            # plt.ylabel("Loss")
            # plt.legend()
            ## this part code relates to interactive training and 
            ## visualising the losses updating
            ## if you want to see the work of it, use the jupyter notebook 
            ## the path to it is: `notebooks/StackGAN.ipynb`
            
            
            # display(plt.gcf())
            if i % 100 == 0:
                inputs = (txt_embedding, fixed_noise)
                lr_fake, fake, _, _ = netG(*inputs)
                save_img_results(real_img_cpu, fake, epoch, image_dir, vis_count=10)
                if lr_fake is not None:
                    save_img_results(None, lr_fake, epoch, image_dir, vis_count=10)
            end_t = time.time()
            print(f'[{epoch}/{max_epoch}][{i}/{len(data_loader)}] Loss_D: {errd.item():.4f} Loss_G: {errg.item():.4f} Loss_KL: {kl_loss.item():.4f}\nLoss_real: {errd_real:.4f} Loss_wrong:{errd_wrong:.4f} Loss_fake {errd_fake:.4f}\nTotal Time: {(end_t - start):.2f}sec')
            if epoch % snapshot_interval == 0:
                save_model(netG, netD, epoch, "model_weights")
                
    torch.save(netG.state_dict(), f'{model_dir}/netG_epoch_last.pth')
    torch.save(netD.state_dict(), f'{model_dir}/netD_epoch_last.pth')
    print('Save G/D models')



image_transform = transforms.Compose([
transforms.Resize((image_size, image_size)),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ds = StackGANDataset(data_path = args.data_path,
                     embedding_path = args.embedding_path, transform=image_transform)        
print(f"sample: {ds.__getitem__(0)}")


dl = torch.utils.data.DataLoader(ds,
                                 batch_size = batch_size,
                                 shuffle = True,
                                 num_workers = 0,
                                 pin_memory = True, drop_last = True)


if __name__ == "__main__":
    train(dl, stage = args.stage)
    
    

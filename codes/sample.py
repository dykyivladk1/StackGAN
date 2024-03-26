import torch
from transformers import CLIPProcessor, CLIPModel

import numpy as np
from PIL import Image

import argparse
import yaml
from polip import decider

from model import *

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cpu")


parser = argparse.ArgumentParser(
    description = "Evaluate StackGAN"
)
parser.add_argument("--yaml_path", type = str, default = "cfg/evaluate.yaml")
parser.add_argument("--prompt", type = str, default = "hello world")
args = parser.parse_args()


with open(args.yaml_path, "r") as file:
    config = yaml.safe_load(file)
    
gf_dim = int(config["model_params"]["gf_dim"])
z_dim = int(config["model_params"]["z_dim"])
text_dim = int(config["model_params"]["text_dim"])
df_dim = int(config["model_params"]["df_dim"])
condition_dim = int(config["model_params"]["condition_dim"])
r_num = int(config["model_params"]["r_num"])

def sample(batch_size, text_prompt: str ,stage = 1):
    if stage == 1:
        netG, netD = load_nns_stage1(STAGE1_G, STAGE1_D, gf_dim,
                                     df_dim, condition_dim, z_dim, text_dim, device)
    else:
        netG, netD = load_nns_stage2(STAGE1_G, STAGE2_G, STAGE2_D,
                                     gf_dim, condition_dim, z_dim, r_num, df_dim, text_dim, device)
    netG.eval()
    inputs = processor(
        text = text_prompt,
        return_tensors = "pt",
        padding = True,
        truncation = True, max_length = 77
    )
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    embeddings = text_embedding.squeeze(0)
    num_embeddings = len(embeddings)
    batch_size = np.minimum(num_embeddings, batch_size)
    noise = torch.randn((batch_size, 100), dtype = torch.float32)
    count = 0
    while count < num_embeddings:
        if count > 3000:
            break
        iend = count + batch_size
        if iend > num_embeddings:
            iend = num_embeddings
            count = num_embeddings - batch_size
        embeddings_batch = embeddings[count:iend]
        txt_embedding = torch.tensor(embeddings_batch, dtype = torch.float32)
        txt_embedding = txt_embedding.to(device).squeeze(0)
        
        noise.data.normal_(0, 1)
        _, fake_imgs, mu, logvar = \
            netG(text_embedding, noise)
        for i in range(batch_size):
            save_name = '%s/%d.png' % ("gen_samples", count + i)
            im = fake_imgs[i].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            im.save(save_name)
            count += batch_size
            
sample(batch_size = 1, text_prompt = args.prompt)

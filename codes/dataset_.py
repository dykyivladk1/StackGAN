import torch
import torchvision.transforms as transforms
from glob import glob
import pickle
from PIL import Image


class StackGANDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, embedding_path, transform = None):
        super().__init__()
        
        self.data_path = data_path
        self.transform = transform
        self.embedding_path = embedding_path
        
        self.images = glob(self.data_path + "/*.jpeg")
        self.embeddings_data = self.load_embeddings()
    def load_embeddings(self):
        with open(self.embedding_path, "rb") as file:
            data = pickle.load(file)
        return data
    def __getitem__(self, idx):
        filepath = self.images[idx]
        image_filename = Image.open(self.images[idx])
        image_filename = image_filename.convert("RGB")
        
        tmp = filepath.split("/")[-1]
        tmp = tmp.replace("jpeg", "txt")
        
        embedding = self.embeddings_data[tmp]
        embedding = embedding[:, :].squeeze(0)
        
        if self.transform is not None:
            image_filename = self.transform(image_filename)
        # embedding = torch.tensor(embedding, dtype = torch.float32)       
        return image_filename, embedding,  filepath
    def __len__(self):
        return len(self.images)
    


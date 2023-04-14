from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import cv2
import torchvision
from torchvision import transforms
import glob
import numpy as np
import torch
import config as conf


class FrameInterpolationDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = list(glob.glob(img_paths +  f'/*.jpg'))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')/255.0

        frame_n = img[:, 0:256]
        frame_n1 = img[:, 256:512]
        frame_n2 = img[:, 512:768]

        input_img = np.concatenate([frame_n,frame_n2],2)

        target_img = frame_n1
        input_img = self.transform(input_img)
        target_img = self.transform(frame_n1)
        return input_img,target_img
    
class RefineDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = list(glob.glob(img_paths +  f'/*.jpg'))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')/255.0

        target = img[:, 0:256]
        input = img[:, 256:512]

        input_img = self.transform(input)
        target_img = self.transform(target)
        return input_img,target_img

def test():
    input_path = 'dataset/train/'
    dataset = FrameInterpolationDataset(input_path)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        # cv2.imshow('framen',np.uint8(x.cpu().numpy()[0].reshape(256,256,6)*255))
        cv2.imshow('framen2',np.uint8(y.squeeze().permute(1, 2, 0).numpy()*255))
        cv2.waitKey(0)
        break

if __name__ == "__main__":
    test()
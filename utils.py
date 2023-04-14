import cv2
import numpy as np
from torchvision import transforms
from IPython.display import clear_output, display
from PIL import Image
import config
import torch


def generate_images(generator, input_path):
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')/255.0

    frame_n = img[:, :256]
    frame_n1 = img[:, 256:512]
    frame_n2 = img[:, 512:]
    # frame_n = frame_n.astype('float32')/255.0
    # frame_n1 = frame_n1.astype('float32')/255.0
    # frame_n2 = frame_n2.astype('float32')/255.0
    

    input_img = np.concatenate([frame_n,frame_n2],2)

    input_tensor = torch.from_numpy(input_img.transpose((2, 0, 1))).unsqueeze(0)
    input_tensor = input_tensor.to(config.DEVICE)

    gen_output = generator(input_tensor).detach().cpu().squeeze().permute(1, 2, 0).numpy()
    
    gen_output = np.concatenate((img, gen_output), axis=1)

    
    # Display the combined image
    display(Image.fromarray(np.uint8(gen_output.clip(0, 1)*255.0)))
    clear_output(wait=True)


def generate_image(generator, img):
    img_shape = img.shape[:2]
    input_img = img.astype('float32')/255.0

    input_tensor = torch.from_numpy(input_img.transpose((2, 0, 1))).unsqueeze(0)
    input_tensor = input_tensor.to(config.DEVICE)

    gen_output = generator(input_tensor).detach().cpu().squeeze().permute(1, 2, 0).numpy()

    gen_output = np.uint8(gen_output.clip(0, 1)*255.0)
    gen_output = cv2.resize(gen_output,(img_shape[1],img_shape[0]))

    return gen_output

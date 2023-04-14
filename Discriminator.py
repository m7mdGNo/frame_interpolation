import torch
import torch.nn as nn
from Generator import Downsample

class Discriminator(nn.Module):
    def __init__(self,input_channels,features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initial_down = nn.Sequential(
            nn.Conv2d(input_channels*3, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Downsample(features,features*2) 
        self.down2 = Downsample(features*2,features*4) 
        self.down3 = Downsample(features*4,features*8) 
        self.down4 = Downsample(features*8,features*8,padding_mode='zeros') 
        self.down5 = Downsample(features*8,features*8,padding_mode='zeros') 
        self.down6 = Downsample(features*8,features*8,padding_mode='zeros')
        self.fc1 = nn.Linear(512*4*4,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1)
    def forward(self,x,y):
        x = torch.cat([x,y],1)  #(9x256x256)
        x = self.initial_down(x) # 128
        x = self.down1(x)   #64
        x = self.down2(x)   #32
        x = self.down3(x)   #16
        x = self.down4(x)   #8
        x = self.down6(x)   #4
        x = x.view(-1, 512*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        return x
    

class Refine_Discriminator(nn.Module):
    def __init__(self,input_channels,features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initial_down = nn.Sequential(
            nn.Conv2d(input_channels*2, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Downsample(features,features*2) 
        self.down2 = Downsample(features*2,features*4) 
        self.down3 = Downsample(features*4,features*8) 
        self.down4 = Downsample(features*8,features*8,padding_mode='zeros') 
        self.down5 = Downsample(features*8,features*8,padding_mode='zeros') 
        self.down6 = Downsample(features*8,features*8,padding_mode='zeros')
        self.fc1 = nn.Linear(512*4*4,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1)
    def forward(self,x,y):
        x = torch.cat([x,y],1)  #(9x256x256)
        x = self.initial_down(x) # 128
        x = self.down1(x)   #64
        x = self.down2(x)   #32
        x = self.down3(x)   #16
        x = self.down4(x)   #8
        x = self.down6(x)   #4
        x = x.view(-1, 512*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        return x
    
def test():
    x = torch.randn((1,6,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator(input_channels=3, features=64)
    preds = model(x,y)
    print(preds[0])
    print(preds.shape)


if __name__ == "__main__":
    test()
import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=4,dropout=False,strides=2,padding=1,padding_mode='reflect', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding
        self.karnel_size = kernel_size
        self.dropout = dropout
        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels,self.output_channels,self.karnel_size,self.strides,self.padding,padding_mode=padding_mode),
            nn.BatchNorm2d(self.output_channels),
            nn.LeakyReLU(0.2)
        )
        if self.dropout:
            self.model.add_module('dropout',nn.Dropout(0.5))
    
    def forward(self,x):
        return self.model(x)
    
class Upsample(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=4,dropout=False,strides=2,padding=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding
        self.karnel_size = kernel_size
        self.dropout = dropout
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.input_channels,self.output_channels,self.karnel_size,self.strides,self.padding),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU()
        )
        if self.dropout:
            self.model.add_module('dropout',nn.Dropout(0.5))
    
    def forward(self,x):
        return self.model(x)
    

class Generator(nn.Module):
    def __init__(self,input_channels,output_channels,features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.initial_down = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Downsample(features,features*2)
        self.down2 = Downsample(features*2,features*4) 
        self.down3 = Downsample(features*4,features*8)
        self.down4 = Downsample(features*8,features*8)
        self.down5 = Downsample(features*8,features*8)
        self.down6 = Downsample(features*8,features*8)


        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        
        self.up1 = Upsample(features*8,features*8,dropout=True)
        self.up2 = Upsample(features*8*2,features*8,dropout=True)
        self.up3 = Upsample(features*8*2,features*8,dropout=True)
        self.up4 = Upsample(features*8*2,features*8)
        self.up5 = Upsample(features*8*2,features*4)
        self.up6 = Upsample(features*4*2,features*2)
        self.up7 = Upsample(features*2*2,features)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2,output_channels,4,2,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        d1 = self.initial_down(x) 
        d2 = self.down1(d1) 
        d3 = self.down2(d2) 
        d4 = self.down3(d3) 
        d5 = self.down4(d4) 
        d6 = self.down5(d5) 
        d7 = self.down6(d6) 

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        final = self.final(torch.cat([up7, d1], 1))
        return final
    

class Refine_Generator(nn.Module):
    def __init__(self,input_channels,output_channels,features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.initial_down = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Downsample(features,features*2)
        self.down2 = Downsample(features*2,features*4) 
        self.down3 = Downsample(features*4,features*8)
        self.down4 = Downsample(features*8,features*8)
        self.down5 = Downsample(features*8,features*8)
        self.down6 = Downsample(features*8,features*8)


        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        
        self.up1 = Upsample(features*8,features*8,dropout=True)
        self.up2 = Upsample(features*8*2,features*8,dropout=True)
        self.up3 = Upsample(features*8*2,features*8,dropout=True)
        self.up4 = Upsample(features*8*2,features*8)
        self.up5 = Upsample(features*8*2,features*4)
        self.up6 = Upsample(features*4*2,features*2)
        self.up7 = Upsample(features*2*2,features)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2,output_channels,4,2,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        d1 = self.initial_down(x) 
        d2 = self.down1(d1) 
        d3 = self.down2(d2) 
        d4 = self.down3(d3) 
        d5 = self.down4(d4) 
        d6 = self.down5(d5) 
        d7 = self.down6(d6) 

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        final = self.final(torch.cat([up7, d1], 1))
        return final

def test():
    x = torch.randn((1, 6, 256, 256))
    model = Generator(input_channels=6,output_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()

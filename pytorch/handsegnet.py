import torch
import torch.nn as nn
from torchsummary import summary

class Conv3x3(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(inp_dim, out_dim, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class HandSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            Conv3x3(3, 64),
            Conv3x3(64, 64),
            nn.MaxPool2d(4, 2, 1)
        )
        self.layer2 = nn.Sequential(
            Conv3x3(64, 128),
            Conv3x3(128, 128),
            nn.MaxPool2d(4, 2, 1)
        )
        self.layer3 = nn.Sequential(
            Conv3x3(128, 256),
            Conv3x3(256, 256),
            Conv3x3(256, 256),
            Conv3x3(256, 256),
            nn.MaxPool2d(4, 2, 1)
        )
        self.layer4 = nn.Sequential(
            Conv3x3(256, 512),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            nn.Conv2d(512, 2, 1),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
        
if __name__ == '__main__':
    device = torch.device('cuda')
    net = HandSegNet().to(device)
    summary(net, (3, 256, 256))
    x = torch.randn((1, 3, 256, 256)).to(device)
    out = net(x)
    print(out.shape)
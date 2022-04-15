import torch 
import torch.nn as nn 

class CustomizedAlexNet(nn.Module):
    def __init__(self):
        super(CustomizedAlexNet, self).__init__()
        # L1 ImgIn shape  = (B, 18, 65, 65)
        #    Conv         = (B, 64, 65, 65)                  
        #    Pool         = (B, 64, 32, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # L2 ImgIn shape = (B, 64, 32, 32) 
        #    Conv        = (B, 128, 32, 32)
        #    Pool        = (B, 128, 16, 16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # L3 ImgIn shape = (B, 128, 16, 16)
        #    Conv        = (B, 256, 8, 8)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # L4 ImgIn shape = (B, 256, 8, 8)
        #    Conv        = (B, 256, 8, 8)
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # L5 ImgIn shape = (B, 256, 8, 8)
        #    Conv        = (B, 256, 8, 8)
        #    Pool        = (B, 256, 4, 4)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # FC1 4 X 4 X 256 inputs --> 256 outputs
        self.fc1 = nn.Linear(4*4*256, 256, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer6 = nn.Sequential(
            self.fc1,
            nn.Dropout(p=0.25),
            nn.ReLU()
        )
        
        # FC2 256 inputs --> 8 outputs
        self.fc2 = nn.Linear(256, 8, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.layer7 = nn.Sequential(
            self.fc2,
            nn.Dropout(p=0.25),
            nn.ReLU()
        )
        
        # FC3 8 inputs --> 1 output
        self.fc3 = nn.Linear(8, 1, bias=True)
        nn.init.xavier_uniform_(self.fc3.weight)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out).view(out.size(0),-1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.fc3(out)
        
        out = torch.sigmoid(out).view(-1)
        
        return out 
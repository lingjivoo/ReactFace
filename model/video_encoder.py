import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_planes=3, planes=128):
        super(ConvBlock, self).__init__()

        self.planes = planes
        self.conv1 = nn.Conv3d(in_planes, planes//4, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)
        self.in1 = nn.InstanceNorm3d(planes//4)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))

        self.conv2 = nn.Conv3d(planes//4, planes, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)
        self.in2 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)


        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        self.in3 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.in4 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv3d(planes, planes, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        self.in5 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.in2(self.conv2(x)))


        if x.shape[-1] != 28:
            x = F.interpolate(x, size = (x.shape[2], 28,28), mode='nearest')

        x = self.relu(self.in3(self.conv3(x)))
        x = x + self.relu(self.in4(self.conv4(x)))
        x = self.relu(self.in5(self.conv5(x)))

        b,c,t,h,w = x.shape
        x = x.mean(dim=-1).mean(dim=-1)
        x = x.transpose(1,2)
        return x


class VideoEncoder(nn.Module):
    def __init__(self, img_size=224, feature_dim = 256, device = 'cpu'):
        super(VideoEncoder, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vector: (batch_size, seq_len, V*3)
        """

        self.img_size = img_size
        self.feature_dim = feature_dim

        self.Conv3D = ConvBlock(3, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.device = device


    def forward(self, video):
        # def forward(self, audio, vertice, one_hot, criterion,teacher_forcing=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        # obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        # video: (B,T,C,H,W)
        # audio: (B,A)

        video_input = video.transpose(1, 2)  # B C T H W
        video_output = self.Conv3D(video_input)
        video_output = self.fc(video_output) # B T C
        return  video_output







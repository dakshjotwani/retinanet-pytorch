import torch
import torch.nn as nn

def lateral_conv2d(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

def conv_3x3(channels):
    return nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

def conv_3x3_stride_2(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

class FeaturePyramid(nn.Module):
    def __init__(self, C3_channels, C4_channels, C5_channels, out_channels=256):
        super(FeaturePyramid, self).__init__()
        
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Up from C5 -> P6 -> P7
        self.P6_up_conv  = conv_3x3_stride_2(C5_channels, out_channels)
        self.P7_up_conv  = conv_3x3_stride_2(out_channels, out_channels)

        # First lateral connection from C5 -> P5
        self.P5_lat_conv = lateral_conv2d(C5_channels, out_channels)
        self.P5_3x3      = conv_3x3(out_channels)

        # Down from P5 -> P4 -> P3
        # Lateral:     ^     ^
        #             C4    C3
        self.P4_lat_conv = lateral_conv2d(C4_channels, out_channels)
        self.P4_3x3      = conv_3x3(out_channels)

        self.P3_lat_conv = lateral_conv2d(C3_channels, out_channels)
        self.P3_3x3      = conv_3x3(out_channels)

    def forward(self, C3, C4, C5):
        # Up
        P6 = self.P6_up_conv(C5)
        P7 = self.P7_up_conv(self.relu(P6))
        
        # Lateral + Down
        P5 = self.P5_lat_conv(C5)
        P4 = self.P4_lat_conv(C4) + self.upsample(P5)
        P3 = self.P3_lat_conv(C3) + self.upsample(P4)

        # Apply 3x3
        P5 = self.P5_3x3(P5)
        P4 = self.P4_3x3(P4)
        P3 = self.P3_3x3(P3)

        return P3, P4, P5, P6, P7
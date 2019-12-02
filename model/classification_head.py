import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(x))
        out = self.relu(self.conv3(x))
        out = self.relu(self.conv4(x))

        out = self.sigmoid(self.output(out))
        
        return out

        # out is B x C x H x W, with C = n_classes + n_anchors
        #out1 = out.permute(0, 2, 3, 1)

        #batch_size, height, width, channels = out1.shape

        #out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        #return out2 #.contiguous().view(x.shape[0], -1, self.num_classes)
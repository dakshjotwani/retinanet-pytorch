import torch
import torch.nn as nn

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import resnet50

from .feature_pyramid import FeaturePyramid
from .classification_head import ClassificationHead
from .regression_head import RegressionHead

class RetinaNet(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super(RetinaNet, self).__init__()
        
        # Set resnet50 backbone and remove instance variables that we won't use
        self.backbone = resnet50(pretrained=pretrained_backbone)
        del self.backbone.avgpool
        del self.backbone.fc
        
        self.feature_pyramid = FeaturePyramid(512,  # Resnet layer 2 (C3) out channels
                                              1024, # Resnet layer 3 (C4) out channels
                                              2048, # Resnet layer 4 (C5) out channels
                                              out_channels=256)

        self.class_head = ClassificationHead(256)
        self.reg_head   = RegressionHead(256)

    def _resnet_backbone_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        C3 = self.backbone.layer2(x)
        C4 = self.backbone.layer3(C3)
        C5 = self.backbone.layer4(C4)

        return C3, C4, C5

    def forward(self, x):
        out = self._resnet_backbone_forward(x)

        # out = (C3, C4, C5)
        out = self.feature_pyramid(*out)

        # out = (P3, P4, P5, P6, P7)
        class_out = [self.class_head(P_i) for P_i in out]
        reg_out   = [self.reg_head(P_i) for P_i in out]

        return class_out, reg_out
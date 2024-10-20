import torch.nn as nn
import torch
from .modules import resnet50

class gaze_network(nn.Module):
    def __init__(self, in_stride=2):
        super(gaze_network, self).__init__()
        self.gaze_network_face = resnet50(pretrained=True, in_stride=in_stride)
        self.gaze_network_eye = resnet50(pretrained=True, in_stride=in_stride)
        self.gaze_fc = nn.Sequential(
            nn.Linear(2048*3, 2)
        )

    def forward(self, face, left, right):
        feature_face = self.gaze_network_face(face)
        feature_left = self.gaze_network_eye(left)
        feature_right = self.gaze_network_eye(right)
        feature = torch.cat((feature_face, feature_left, feature_right), 1)
        feature = feature.view(feature.size(0), -1)

        gaze = self.gaze_fc(feature)
        return gaze

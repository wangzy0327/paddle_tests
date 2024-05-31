import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

class FaceNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = ResNet(BottleneckBlock, 50, num_classes=128)
    
    def l2_normalize(self, x, axis=-1):
        return x / paddle.sqrt(paddle.sum(x**2, axis, keepdim=True))
    
    def forward(self, x):
        x = self.model(x)
        x = F.normalize(x, p=2, axis=-1)
        return x

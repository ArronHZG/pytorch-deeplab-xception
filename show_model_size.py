from torchsummary import summary
from tensorboardX import SummaryWriter

from modeling.deeplab import *
import torch

writer = SummaryWriter(f'./logs/tensorBoardX/2019-07-09--16:38:27')
# 加载模型
model = DeepLab(backbone='resnet_4c',output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False)
model_name = 'DeepLa-resnet_4c'
# 模型打印
summary(model, (4,256,256),device="cpu")
# model可视化
x = torch.rand(1,4,256,256)  # 随便定义一个输入
model.freeze_bn()
writer.add_graph(model, x)
writer.close()
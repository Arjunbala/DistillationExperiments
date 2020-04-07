'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, index=0):
        super(BasicBlock, self).__init__()
        self.index = index
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        start = time.perf_counter()
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #torch.cuda.synchronize()
        end=time.perf_counter()
        print("Index: " + str(self.index) + " Time(ms): {:.04}".format((end-start)*1000/2))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, index=0):
        super(Bottleneck, self).__init__()
        self.index = index
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        print(x.view(-1).shape)
        start = time.perf_counter()
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        torch.cuda.synchronize()
        end=time.perf_counter()
        print("Index: " + str(self.index) + " Time(ms): {:.04}".format((end-start)*1000))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, index=0)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, index=num_blocks[0])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, index=num_blocks[0]+num_blocks[1])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, index=num_blocks[0]+num_blocks[1]+num_blocks[2])
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, index):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        running_index = index
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, index=running_index))
            running_index = running_index + 1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        start=time.perf_counter()
        out = F.relu(self.bn1(self.conv1(x)))
        torch.cuda.synchronize()
        end=time.perf_counter()
        print("Before : Time(ms): {:.04}".format((end-start)*1000))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        start=time.perf_counter()
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        torch.cuda.synchronize()
        end=time.perf_counter()
        print("After : Time(ms): {:.04}".format((end-start)*1000))
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152(num_classes = 10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)


def test():
    os.environ["OMP_NUM_THREADS"] = '8'
    net = ResNet50().cuda()
    for i in range(0,100):
        start=time.perf_counter()
        y = net(torch.randn(1,3,32,32).cuda())
        end=time.perf_counter()
        print("Total : Time(ms): {:.04}".format((end-start)*1000))

test()

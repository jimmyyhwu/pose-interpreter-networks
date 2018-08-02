import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.resnet18 = resnet18(pretrained=False, num_input_channels=cfg.num_input_channels)

        num_input_nodes = 512 * 8 * 10

        self.num_shared_fc_layers = cfg.num_shared_fc_layers
        if self.num_shared_fc_layers > 0:
            for i in range(self.num_shared_fc_layers):
                num_output_nodes = cfg.num_shared_fc_nodes
                setattr(self, 'fc{}'.format(i + 1), nn.Linear(num_input_nodes, num_output_nodes))
                num_input_nodes = num_output_nodes
        num_shared_output_nodes = num_input_nodes

        self.num_position_fc_layers = cfg.num_position_fc_layers
        self.num_orientation_fc_layers = cfg.num_orientation_fc_layers
        assert self.num_position_fc_layers > 0
        assert self.num_orientation_fc_layers > 0

        for i in range(self.num_position_fc_layers):
            if i == self.num_position_fc_layers - 1:
                num_output_nodes = cfg.num_position_outputs
            else:
                num_output_nodes = cfg.num_position_fc_nodes
            setattr(self, 'fc_p{}'.format(i + 1), nn.Linear(num_input_nodes, num_output_nodes))
            num_input_nodes = num_output_nodes

        num_input_nodes = num_shared_output_nodes
        for i in range(self.num_orientation_fc_layers):
            if i == self.num_orientation_fc_layers - 1:
                num_output_nodes = cfg.num_orientation_outputs
            else:
                num_output_nodes = cfg.num_orientation_fc_nodes
            setattr(self, 'fc_o{}'.format(i + 1), nn.Linear(num_input_nodes, num_output_nodes))
            num_input_nodes = num_output_nodes

    def forward(self, x, object_index):
        x = self.resnet18(x)
        x = x.view(-1, 512 * 8 * 10)

        for i in range(self.num_shared_fc_layers):
            x = self.relu(getattr(self, 'fc{}'.format(i + 1))(x))

        p = x
        for i in range(self.num_position_fc_layers - 1):
            p = self.relu(getattr(self, 'fc_p{}'.format(i + 1))(p))
        p = getattr(self, 'fc_p{}'.format(self.num_position_fc_layers))(p)

        o = x
        for i in range(self.num_orientation_fc_layers - 1):
            o = self.relu(getattr(self, 'fc_o{}'.format(i + 1))(o))
        o = getattr(self, 'fc_o{}'.format(self.num_orientation_fc_layers))(o)

        # select appropriate output based on object index
        p = p.view(p.size(0), -1, 3)
        o = o.view(p.size(0), -1, 4)
        object_index_view = object_index.view(object_index.size(0), 1, 1)
        p = torch.gather(p, 1, object_index_view.expand(object_index_view.size(0), 1, 3)).squeeze(1)
        o = torch.gather(o, 1, object_index_view.expand(object_index_view.size(0), 1, 4)).squeeze(1)
        q_norm = o.norm(dim=1, keepdim=True)
        o = o.div(q_norm)

        return p, o


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_input_channels=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        model.load_state_dict(state_dict)
    return model

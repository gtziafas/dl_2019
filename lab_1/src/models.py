import math
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import models, datasets, transforms


#the GELU activation function
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

#define model hyperparameters
batch_size = 256
epochs = 12
learning_rate = .01
img_crop_size = 64
print_step = 120
weight_decay = 0.01

global activation
lookup_act = {
	'ReLU'		 :	nn.ReLU(inplace=True),
	'ELU'		 :	nn.ELU(),
	'lRELU'		 :	nn.LeakyReLU(),
	'GeLU'		 :  GELU()	
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        global activation
        self.relu = activation
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                 **kwargs)


#define ResNet architecture
def resnet(activ='ReLU', optim='SGD+mom'):
	activ = 'ReLU'
	global activation
	activation = lookup_act[activ]
	model = resnet18(activation, num_classes=10)

	print("Using model: \n", model)
	optim = 'SGD+mom'
	#use gpu tensors if available
	device = 'cpu'
	if torch.cuda.is_available():
	    model = model.cuda()
	    device = 'cuda'
	    print('Using cuda.\n')
	else:
	    print('Using cpu.\n')

	#data preproc stage - img format: {batch_size X 3 X img_size X img_size}
	#convert to Tensor objects and normalize to floats in [0,1]
	transform = transforms.Compose([
		transforms.Resize(img_crop_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	#define dataset - here CIFAR10
	train_data = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
	test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform)

	#shuffle and batch data inside DataLoader objects
	trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
	testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

	#define loss function and optimization algorithm
	loss_fn = nn.CrossEntropyLoss()	#here cross-entropy for multiclass classficiation

	if optim == 'SGD+mom':
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.93, weight_decay=weight_decay)
	elif optim == 'Adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=weight_decay)
	elif optim == 'RMSprop':
		optimizer = torch.optim.RMSprop(model.parameters(), lr=.0001, alpha=.95, weight_decay=weight_decay) 
	else:
		raise ValueError

	#train the model on the train set, while validating on the validation set 
	train_losses, eval_losses = train(model, trainloader, testloader, optimizer, loss_fn, epochs, learning_rate, device)
	#make predictions for a test set
	accuracy = test(model, trainloader, loss_fn, device)
	print("Model accuracy on train set: %.1f %%" % accuracy)
	accuracy = test(model, testloader, loss_fn, device)
	print("Model accuracy on test set: %.1f %%" % accuracy)
	plt.title('AlexNet training - activation functions (Adam)') 
	plt.xlabel('epochs')
	plt.ylabel('cross-entropy loss')
	plt.plot(train_losses, '-b', label='train')
	plt.plot(eval_losses, '-bv', label='test')
	plt.show()
	
	return train_losses, eval_losses


#define AlexNet Architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes, activation):
        super(AlexNet, self).__init__()
        self.activation = self.lookup(activation)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            self.activation,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.activation,
            nn.Linear(4096, num_classes),
        )

    # takes in a module and applies the specified weight initialization
    def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def lookup(self, f): #activation function wrapping
        if f=='ReLU':
            return nn.ReLU(inplace=True)
        elif f=='ELU':
            return nn.ELU()
        elif f=='lReLU':
            return nn.LeakyReLU(negative_slope=0.01)
        elif f=='GeLU':
            return GELU()

def alexnet(activation='ReLU', optim='SGD+mom'):
    #if not sys.argv[1:]:
    #   print('\nGive arguements. Usage:\n')
    #   return 0
    #else:
    activation = 'ReLU'
    optim='SGD+mom'
    model = AlexNet(num_classes=10, activation=activation)
    #model.weights_init_uniform()

    print("Using model: \n", model)

    #use gpu tensors if available
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')

    #data preproc stage - img format: {batch_size X 3 X img_size X img_size}
    #convert to Tensor objects and normalize to floats in [0,1]
    transform = transforms.Compose([
        transforms.Resize(img_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #define dataset - here CIFAR10
    train_data = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform)

    #shuffle and batch data inside DataLoader objects
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    #define loss function and optimization algorithm
    loss_fn = nn.CrossEntropyLoss() #here cross-entropy for multiclass classficiation
    if optim == 'SGD+mom':
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.93, weight_decay=weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=weight_decay)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=.001, alpha=.95, weight_decay=weight_decay) 
    else:
        raise ValueError

    #train the model on the train set, while validating on the validation set 
    train_losses, eval_losses = train(model, trainloader, testloader, optimizer, loss_fn, epochs, 
                            learning_rate, device)
    #make predictions for a test set
    accuracy = test(model, trainloader, loss_fn, device)
    print("Model accuracy on train set: %.1f %%" % accuracy)
    accuracy = test(model, testloader, loss_fn, device)
    print("Model accuracy on test set: %.1f %%" % accuracy)
    plt.title('AlexNet training - activation functions (Adam)') 
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy loss')
    plt.plot(train_losses, '-b', label='tran')
    plt.plot(eval_losses, '-bv', label='test')
    plt.legend()
    plt.show()
    return train_losses, eval_losses

class VGG(nn.Module):

    def __init__(self, features, activation, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.activation = lookup_act[activation]
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            self.activation,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.activation,
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, activation, **kwargs):   
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), activation, **kwargs)

def vgg16(activation, pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, activation, **kwargs)


def vgg16_bn(activation, pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, activation, **kwargs)

#define ResNet architecture
def vgg(activation='ReLU', optim='SGD+mom'):

    activation = 'ReLU'
    optim = 'SGD+mom'
    model = vgg16_bn(activation, num_classes=10)
    print("Using model: \n", model)

    #get the working directory where the data is saved
    import os
    datadir = os.getcwd()

    #use gpu tensors if available
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')

    #data preproc stage - img format: {batch_size X 3 X img_size X img_size}
    #convert to Tensor objects and normalize to floats in [0,1]
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #define dataset - here CIFAR10
    train_data = datasets.CIFAR10(root='./data/', train=True, download=False, transform=transform)
    test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform)

    #shuffle and batch data inside DataLoader objects
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    #define loss function and optimization algorithm
    loss_fn = nn.CrossEntropyLoss() #here cross-entropy for multiclass classficiation
    if optim == 'SGD+mom':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.93, weight_decay=weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=weight_decay)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=.0001, alpha=.95, weight_decay=weight_decay) 
    else:
        raise ValueError

        #train the model on the train set, while validating on the validation set 
    train_losses, eval_losses = train(model, trainloader, testloader, optimizer, loss_fn, epochs, learning_rate, device)
    #make predictions for a test set
    accuracy = test(model, trainloader, loss_fn, device)
    print("Model accuracy on train set: %.1f %%" % accuracy)
    accuracy = test(model, testloader, loss_fn, device)
    print("Model accuracy on test set: %.1f %%" % accuracy)
    plt.title('VGG training - activation functions')
    plt.ylabel('cross-entropy loss')
    plt.plot(train_losses, '-b', label='train')
    plt.plot(eval_losses, '-bv', label='test')
    plt.show()
    
    return train_losses, eval_losses


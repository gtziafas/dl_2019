import math
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import models, datasets, transforms

#define model hyperparameters
batch_size = 256
epochs = 20
learning_rate = .01
img_crop_size = 64
dropout_rate = 0
weight_decay = .01

#the GELU activation function
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

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
	#	print('\nGive arguements. Usage:\n')
	#	return 0
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
	loss_fn = nn.CrossEntropyLoss()	#here cross-entropy for multiclass classficiation
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


if __name__ == '__main__':
    alexnet(sys.argv)
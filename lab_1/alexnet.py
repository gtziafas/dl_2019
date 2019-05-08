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
			#nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			self.activation,
			#nn.Dropout(),
			nn.Linear(4096, 4096),
			self.activation,
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x

	def lookup(self, f): #activation function wrapping
		if f=='ReLU':
			return nn.ReLU(inplace=True)
		elif f=='PReLU':
			return nn.PReLU()
		elif f=='lReLU':
			return nn.LeakyReLU()
		elif f=='GeLU':
			return GELU()

#define model hyperparameters
batch_size = 256
epochs = 20
learning_rate = .01
img_size = 64
print_step = 120

def alexnet(activation, optimizer, dropout=False, weight_decay=False):
	#if not sys.argv[1:]:
	#	print('\nGive arguements. Usage:\n')
	#	return 0
	#else:
	model = models.alexnet()
	# Number of filters in the bottleneck layer
	num_ftrs = model.classifier[6].in_features
	# convert all the layers to list and remove the last one
	features = list(model.classifier.children())[:-1]
	## Add the last layer based on the num of classes in our dataset
	features.extend([nn.Linear(num_ftrs, n_class=10)])
	## convert it into container and add it to our model class.
	model.classifier = nn.Sequential(*features)

	print("Using model: \n", model)

	#get the working directory where the data is saved
	import os
	datadir = os.getcwd()

	#use gpu tensors if available
	if torch.cuda.is_available():
	    use_gpu = True
	    model = model.cuda()
	    print('Using cuda.\n')
	else:
	    use_gpu = False
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
	loss_fn = nn.CrossEntropyLoss()	#here cross-entropy for multiclass classficiation
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.93) #here SGD with momentum

	#start training
	print("\nStarted training...\n")
	model.train()
	train_losses = []
	for epoch in range(epochs):
		total = 0
		correct = 0
		for batch_idx, (imgs, labels) in enumerate(trainloader):
			imgs = Variable(imgs)
			labels = Variable(labels)

			if use_gpu:
				imgs = imgs.cuda()
				labels = labels.cuda()
			
			optimizer.zero_grad()
			
			#forward pass and loss
			predictions = model(imgs)
			loss = loss_fn(predictions, labels)

			#back prop and optimization
			loss.backward()
			optimizer.step()

			if (batch_idx+1) % print_step ==0:
				print("Epoch [%d/%d], Batch [%d/%d] Loss: %.4f" % (epoch + 1, epochs, batch_idx + 1, len(trainloader), loss.item()))

			#compute accuracy of this epoch
			_, predicted = torch.max(predictions.data, 1)
			total += labels.size(0)
			correct += (predicted == labels.data).sum()

		print("Model accuracy on the train set at step [%d/%d]: %.1f %%" % (epoch+1, epochs, (100*correct/total)))
		train_losses.append((1-correct/total))

		#weight decay
		NotImplemented

	#test model
	model.eval()
	correct=0
	total=0
	for imgs, labels in testloader:
		imgs = Variable(imgs)
		labels = Variable(labels)

		if use_gpu:
			imgs = imgs.cuda()
			labels = labels.cuda()

		predictions = model(imgs)
		_, predicted = torch.max(predictions.data, 1)
		total += labels.size(0)
		temp = (predicted == labels.data).sum()
		correct += temp

	print("Model accuracy on test set: %.1f %%" % (100*correct/total))
	plt.title('AlexNet training - activation functions')
	plt.xlabel('epochs')
	plt.ylabel('cross-entropy loss')
	plt.legend('ReLU')
	plt.plot(train_losses)
	plt.show()
	return train_losses

if __name__ == '__main__':
    alexnet('ReLU', True, False, False)
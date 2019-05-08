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
epochs = 20
learning_rate = .01
img_size = 64
print_step = 120

#define ResNet architecture
def resnet(activation, optimizer, dropout=False, weight_decay=False):

	model = models.resnet34()

	#change the last layer to output 10 classes
	num_feats = model.fc.in_features
	model.fc  = nn.Linear(num_feats, n_clas=10)
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

	from matplotlib import pyplot as plt
	print("Model accuracy on test set: %.1f %%" % (100*correct/total))
	plt.title('AlexNet training - activation functions')
	plt.xlabel('epochs')
	plt.ylabel('cross-entropy loss')
	plt.legend('ReLU')
	plt.plot(train_losses)
	plt.show()

	return train_losses

if __name__ == '__main__':
    resnet('ReLU', True, False, False)
from alexnet import alexnet
from vgg import vgg 
from resnet import resnet 

from matplotlib import pyplot as plt
import numpy as np 

architectures = ['AlexNet', 'VGG', 'ResNet']
activations = ['ReLU', 'ELU', 'lReLU', 'GeLU']
optimizers = ['SGD+mom', 'RMSprop', 'Adam']

for arch in architectures:
	loss_, acc_ = [], []
	#test different activation functions
	for act in activations:
		print('\nTesting {}: activations\n'.format(arch))
		if not architectures.index(arch):
			l_, a_ = alexnet(activation=act)
		elif architectures.index(arch) == 1:
			l_, a_ = vgg(activation=act)
		else:
			l_, a_ = resnet(activ=act)
		loss_.append([])
		acc_.append([])
		loss_[activations.index(act)] = l_
		acc_[activations.index(act)] = a_
		
	plt.title('{} training - activation functions'.format(arch)) 
	plt.xlabel('epochs')
	plt.ylabel('cross-entropy loss')
	plt.plot([x*10 for x in loss_[0]], '-b', label='ReLU')
	#plt.plot([x*10 for x in acc_[0]], '-bv')
	plt.plot([x*10 for x in loss_[1]], '-g', label='ELU')
	#plt.plot([x*10 for x in acc_[1]], '-gv')
	plt.plot([x*10 for x in loss_[2]], '-r', label='Leaky ReLU')
	#plt.plot([x*10 for x in acc_[2]], '-rv')
	plt.plot([x*10 for x in loss_[3]], '-y', label='GeLU')
	#plt.plot([x*10 for x in acc_[3]], '-yv')
	plt.legend()
	plt.show() 

loss_, acc_ = [], []
for arch in architectures:
	#test different optimizers
	for opt in optimizers:
		print('\nTesting {}: optimizer: {}\n'.format(arch, opt))
		if arch == 'AlexNet':
			l_, a_ = alexnet(optim=opt)
		elif arch == 'VGG':
			l_, a_ = vgg(optim=opt)
		else:
			l_, a_ = resnet(optim=opt)
		loss_.append([])
		acc_.append([])
		loss_[optimizers.index(opt)] = l_
		acc_[optimizers.index(opt)] = a_
	
	print(loss_)
	print(a_)
	plt.title('{} training - optimization algorithms'.format(arch)) 
	plt.xlabel('epochs')
	plt.ylabel('cross-entropy loss')
	plt.plot([x*10 for x in loss_[0]], '-b', label='SGD+mom')
	#plt.plot([x*10 for x in acc_[0]], '-bv')
	plt.plot([x*10 for x in loss_[1]], '-g', label='RMSprop')
	#plt.plot([x*10 for x in acc_[1]], '-gv')
	plt.plot([x*10 for x in loss_[2]], '-r', label='Adam')
	#plt.plot([x*10 for x in acc_[2]], '-rv')
	plt.legend()
	plt.show()



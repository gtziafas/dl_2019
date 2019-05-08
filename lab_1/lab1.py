from alexnet import alexnet
from matplotlib import pyplot as plt

#baseline -> [0],[0]
activations = ['ReLU', 'PReLU', 'LeakyReLU', 'GeLU']
optimizers = ['SGD_mom', 'RMSprop', 'Adam', 'Adamax']

#for AlexNet
#test different activation functions
losses = []
for act in activations:
    print('\nTesting AlexNet: activations\n')
    temp = alexnet(act, optimizers[0])
    losses.append(temp)
plt.title('AlexNet training - activation functions')
plt.xlabel('epochs')
plt.ylabel('cross-entropy loss')
plt.legend('ReLU')
plt.plot(losses)
plt.show()
#test different optimization algorithms
'''
for opt in optimizers:
    print('\nTesting AlexNet: optimizers\n')
    alexnet(activations[0], opt)

#include regularization methods
print('\nTesting AlexNet: regularization\n')
alexnet(act[0], opt[0], True, False) #with dropout
alexnet(act[0], opt[0], True, True)  #with weight decay

#for ResNet
#test different activation functions
for act in activations:
    print('\nTesting AlexNet: activations\n')
    alexnet(act, optimizers[0])

#test different optimization algorithms
for opt in optimizers:
    print('\nTesting AlexNet: optimizers\n')
    alexnet(act[0], opt)

#include regularization methods
print('\nTesting AlexNet: regularization\n')
alexnet(act[0], opt[0], True, False) #with dropout
alexnet(act[0], opt[0], True, True)  #with weight decay

#for Inception
NotImplemented

#for VGG
NotImplemented '''
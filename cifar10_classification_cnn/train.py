import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from CNN import CNN
import dataloader as dl

def main():
	# get CIDAR10 train and test data
	trainloader, testloader, classes = dl.loadCIFAR10Data('./data/CIFAR10')

	# create object of the network
	cnnNet = CNN()
	# load network on gpu
	cnnNet.cuda()

	# define loss function
	criterion = nn.CrossEntropyLoss()
	# define optimizer
	optimizer = optim.SGD(cnnNet.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(10):

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data

			# wrap them in Variable
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

			# zero the parameters gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			# this automatically calls the forward() in CNN class
			outputs = cnnNet(inputs)
			# comput the loss
			loss = criterion(outputs, labels)
			# compute the gradients
			loss.backward()
			# update the weights
			optimizer.step()

			# print statistics
			running_loss += loss.data[0]
			if i % 2000 == 1999:
				print('[%d, %5d] loss: %.3f' % 
					   (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0

	print('Finished training')

	print('\n~~~~~~~ testing network')
	correct = 0
	total = 0

	for data in testloader:
		images, labels = data
		outputs = cnnNet(Variable(images.cuda()))
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	print('Accuracy of the network on the 10000 test images: %d %%'
		  % (100 * correct/total))

	print('\n~~~~~~~~checking accuracy for individual classes')
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))

	for data in testloader:
		images, labels = data
		outputs = cnnNet(Variable(images.cuda()))
		_, predicted = torch.max(outputs.data, 1)
		c = (predicted == labels).squeeze()

		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i]
			class_total[label] += 1

	for i in range(10):
		print('Accuracy of %5s: %2d %%' % (
			  classes[i], 100 * class_correct[i]/class_total[i]))

if __name__ == '__main__':
	main()
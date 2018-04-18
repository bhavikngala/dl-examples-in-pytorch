import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# every class inherits the neural network module
class CNN(nn.Module):

	def __init__(self):
		# call the super function
		super(CNN, self).__init__()

		# define the layers
		self.conv1 = nn.Conv2d(
			in_channels=3,
			out_channels=6,
			kernel_size=5)
		self.pool = nn.MaxPool2d(2, stride=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	# define the forward function
	# basically the flow of input throught the network
	# it is used automatically by the network
	def forward(self, x):
		# pass through conv1 layer, apply relu
		# then pass through pool layer
		x = self.pool(F.relu(self.conv1(x)))
		# pass through conv2 layer, apply relu
		# then pass through pool layer
		x = self.pool(F.relu(self.conv2(x)))
		# reshape the conv2 output to pass
		# to the fully connected layer
		x = x.view(-1, 16 * 5 * 5)
		# pass through fc1 layer and apply relu
		x = F.relu(self.fc1(x))
		# pass through fc2 layer and apply relu
		x = F.relu(self.fc2(x))
		# pass through fc3
		x = self.fc3(x)
		return(x)
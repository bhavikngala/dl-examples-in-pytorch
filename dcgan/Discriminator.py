import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel

class Discriminator(nn.Module):

	def __init__(self):

		super(Discriminator, self).__init__()

		# replace with detecting number of gpus automatically
		self.ngpu = 1

		self.main = nn.Sequential(
			# input is (3) * 64 * 64
			nn.Conv2d(
				in_channels = 3,
				out_channels = 64,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),
			nn.LeakyRelu(negative_slope = 0.2, inplace = True),

			# state size. (64) * 32 * 32
			nn.Conv2d(
				in_channels = 64,
				out_channels = 64 * 2,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),
			nn.BatchNorm2d(feature_size = 64 * 2),
			nn.LeakyRelu(negative_slope = 0.2, inplace = True),

			# state size. (64 * 2) * 16 * 16
			nn.Conv2d(
				in_channels = 64 * 2,
				out_channels = 64 * 4,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),
			nn.BatchNorm2d(feature_size = 64 * 4),
			nn.LeakyRelu(negative_slope = 0.2 inplace = True),

			# state size. (64 * 4) * 8 * 8
			nn.Conv2d(
				in_channels = 64 * 4,
				out_channels = 64 * 8,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),
			nn.BatchNorm2d(feature_size = 64 * 8),
			nn.LeakyRelu(negative_slope = 0.2, inplace = True),

			# state size. (64 * 8) * 4 * 4
			nn.Conv2d(
				in_channels = 64 * 8,
				out_channels = 1,
				kernel_size = 4,
				stride = 1,
				padding = 0,
				bias = False
			),
			nn.Sigmoid()
		)

	def forward(self, input):
		if isintance(input.data, torch,cuda,FloatTensor) and self.ngpu > 1:
			# then run parallely on multiple GPUs
			output = nn.parallel.data_parallel(
				self.main,
				input,
				range(self.ngpu)
			)
		else:
			output = self.main(input)

		return output.view(-1, 1).squeese(1)
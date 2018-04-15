import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel

class Generator(nn.Module):

	def __init__(self):
		
		super(Generator, self).__init__()

		# replace with detecting number of gpus automatically
		self.ngpu = 1

		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(
				in_channels = 100, # dimension of the latent variable Z
				out_channels = 64 * 8, # output of the deconv operation
				kernel_size = 4,
				stride = 1,
				padding = 0,
				bias = False

			),
			# applying batch normazation
			nn.BatchNorm2d(num_features = 64 * 8),
			nn.Relu(inplace=True),
			
			# state size. (64 * 8) * 4 * 4
			nn.ConvTranspose2d(
				in_channels = 64 * 8,
				out_channels = 64 * 4,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),
			nn.BatchNorm2d(num_features = 64 * 4),
			nn.Relu(inplace = True),

			# state size. (64 * 4) * 8 * 8
			nn.ConvTranspose2d(
				in_channels = 64 * 4,
				out_channels = 64 * 2,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),
			nn.BatchNorm2d(num_features = 64 * 2),
			nn.Relu(inplace = True),

			# state size. (64 * 2) * 16 * 16
			nn.ConvTranspose2d(
				in_channels = 64 * 2,
				out_channels = 64,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),
			nn.BatchNorm2d(num_features = 64),
			nn.Relu(inplace = True),

			# state size. (64) * 32 * 32
			nn.ConvTranspose2d(
				in_channels = 64,
				out_channels = 3,
				kernel_size = 4,
				stride = 2,
				padding = 1,
				bias = False
			),

			nn.Tanh()
			# state size. (3) * 64 * 64
		)

	def forward(self, input):
		
		if isinstance(input.data, torch.cuda.FloatTensore) and self.ngpu > 1:
			# then run parallely
			output = nn.parallel.data_parallel(
				self.main,
				input,
				range(self.ngpu)
			)
		else:
			output = self.main(input)

		return output
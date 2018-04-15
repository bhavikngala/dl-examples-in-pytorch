import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from Generator import Generator
from Discriminator import Discriminator

# data directories
outputDir = './output'
dataDir = './data'
generatorNetDir = ''
discriminatorNetDir = ''

# create output directory
try:
	os.makedirs(outputDir)
except OSError:
	pass

# read, transform, load data
dataset = dset.LSUN(
	root = dataDir,
	classes = ['bedroom_train'],
	transform = transforms.Compose([
		transforms.Resize(64),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
)

assert dataset
dataloader = torch.utils.data.DataLoader(
	dataset,
	batch_size = 64,
	shuffle = True,
	num_workers = 2
)

# custom weights initialization called on G and D networks
def weights_init(net):
	classname = net.__class__.__name__
	if classname.find('Conv') != -1:
		net.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		net.weight.data.normal_(1.0, 0.02)
		net.bias.data.fill_(0)

# create network object and initialize weights
# Generator
generator = Generator()
generator.apply(weights_init)

## load the network if path provided
if generatorNetDir != '':
	generator.load.state_dict(torch.load(generatorNetDir))

print(generator)

# Discriminator
discriminator = Discriminator()
discriminator.apply(weights_init)

if discriminatorNetDir != '':
	discriminator.load.state_dict(torch.load(discriminatorNetDir))

print(discriminator)

# define the loss function
criterion = nn.BCELosss()

# define tensors
input = torch.FloatTensor(64, 3, 64, 64)
noise = torch.FloatTensor(64, 100, 1, 1) # latent variable Z
fixed_noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
label = torch.FloatTensor(64)
real_label = 1
fake_label = 0

# if GPU is availeble then use GPU
if torch.cuda.is_available():
	print('\n~~~~~~~ copying to GPU')
	generator.cuda()
	discriminator.cuda()
	criterion.cuda()
	input, label = input.cuda(), label.cuda()
	noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(
	discriminator.parameters(),
	lr = 0.0002,
	betas = (0.5, 0.999)
)
optimizerG = optim.Adam(
	generator.parameters(),
	lr = 0.0002,
	betas = (0.5, 0.999)
)

numEpochs = 25
# training
for epoch in range(numEpochs):
	for i, data in enumerate(dataloader, 0):
		###########################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
		# train with real
		discriminator.zero_grad()
		real_cpu, _ = data
		batch_size = real_cpu.size(0)

		if torch.cuda.is_available():
			real_cpu = real_cpu.cuda()

		input = resize_as_(real_cpu).copy_(real_cpu)
		label.resize_(batch_size).fill_(real_label)
		inputv = Variable(input)
		labelv = Variable(label)

		output = discriminator(inputv)
		discriminatorErrorReal = criterion(output, labelv)
		discriminatorErrorReal.backward()
		D_x = output.data.mean()

		# train with fake
		noise.resize_(batch_size, 100, 1, 1).normal_(0, 1)
		noizev = Variable(noize)
		fakeData = generator(noizev)
		labelv = Variable(label.fill_(fake_label))
		output = discriminator(fakeData.detach())
		discriminatorErrorFake = criterion(output, labelv)
		discriminatorErrorFake.backward()
		D_G_z1 = output.data.mean()
		
		# update discriminator parameters
		discriminatorError = discriminatorErrorReal + discriminatorErrorFake
		optimizerD.step()

		###########################
		# (2) Update G network: maximize log(D(G(z)))
		###########################
		generator.zero_grad()
		# fake labels are real for generator cost
		labelv = Variable(label.fill_(real_label))
		output = discriminator(fakeData)
		generatorError = criterion(output, labelv)
		generatorError.backward()
		D_G_z2 = output.data.mean()
		optimizerG.step()

		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f\
			   D(G(z)):%.4f / %.4f' % (epoch, numEpochs, i,
			   len(dataloader), discriminatorError.data[0],
			   generatorError.data[0], D_x, D_G_z1, D_G_z2))

		if i % 100 == 0:
			vutils.save_image(real_cpu,
				'%s/real_samples.png' % outputDir,
				normalize = True)

			fakeData = generator(fixed_noise)
			vutils.save_image(fakeData.data,
				'%s/fake_samples_epoch_%03d.png' % (outputDir, epoch),
				normalize = True)
	# do checkpointing
	torch.save(
		generator.state_dict(),
		'%s/generator_epoch_%03d.pth' % (outputDir, epoch)
	)
	torch.save(
		discriminator.state_dict(),
		'%s/discriminator_epoch_%03d.pth' % (outputDir, epoch)
	)
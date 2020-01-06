import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	""" Convolutional Neural Network
	"""

	def __init__(self, num_classes=10, drop_p=0.4):
		""" Initialise simple Convolutional Neural Network
		Arguments:
		------------------------
			.. num_classes (int) - number of classes/target groups in dataset
			.. drop_p (float) - probability constant for dropout layers
		"""
		super().__init__()

		# First hidden layer
		self.conv2d_0 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
		self.convbn_0 = nn.BatchNorm2d(num_features=64)

		self.conv2d_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
		self.convbn_1 = nn.BatchNorm2d(num_features=64)

		self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.drop_1 = nn.Dropout2d(p=drop_p)

		# Second hidden layer
		self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.convbn_2 = nn.BatchNorm2d(num_features=128)

		self.conv2d_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.convbn_3 = nn.BatchNorm2d(num_features=128)

		self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.drop_2 = nn.Dropout2d(p=drop_p)

		# Third hidden layer
		self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.convbn_4 = nn.BatchNorm2d(num_features=256)
		
		self.conv2d_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.convbn_5 = nn.BatchNorm2d(num_features=256)

		self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.drop_3 = nn.Dropout(p=drop_p)

		# Dense fully connected layer
		self.dense_linear_1 = nn.Linear(256*3*3, 512)
		self.drop_4 = nn.Dropout(p=drop_p)

		self.dense_linear_2 = nn.Linear(512, 256)
		self.drop_5 = nn.Dropout(p=drop_p)

		self.dense_linear_3 = nn.Linear(256, 128)
		self.out_layer = nn.Linear(128, num_classes)

	def forward(self, x):
		""" Control the feed-forward flowArguments:
		------------------------
			.. x (torch.Tensor): 4D tensor of shape [batch_size, channel, width, height]
		"""
		x = self.conv2d_0(x)
		x = self.convbn_0(x)
		x = F.leaky_relu(x)
		
		x = self.conv2d_1(x)
		x = self.convbn_1(x)
		x = F.leaky_relu(x)

		x = self.pool_1(x)
		x = self.drop_1(x)

		x = self.conv2d_2(x)
		x = self.convbn_2(x)
		x = F.leaky_relu(x)

		x = self.conv2d_3(x)
		x = self.convbn_3(x)
		x = F.leaky_relu(x)

		x = self.pool_2(x)
		x = self.drop_2(x)

		x = self.conv2d_4(x)
		x = self.convbn_4(x)
		x = F.leaky_relu(x)
		
		x = self.conv2d_5(x)
		x = self.convbn_5(x)
		x = F.leaky_relu(x)
		
		x = self.pool_3(x)
		x = self.drop_3(x)

		x = x.view(-1, 256*3*3)
		x = self.dense_linear_1(x)
		x = F.relu(x)
		x = self.drop_4(x)
		
		x = self.dense_linear_2(x)
		x = F.relu(x)
		x = self.drop_5(x)
		
		x = self.dense_linear_3(x)
		x = F.relu(x)

		out = self.out_layer(x)
		return out

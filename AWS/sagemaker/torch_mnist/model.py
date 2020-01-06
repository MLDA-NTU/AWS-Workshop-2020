import torch
import torch.nn as nn
import torch.nn.functional as F


class CEncoder(nn.Module):
	""" Encoder module of simple Conditional VAE network
	"""

	def __init__(self, input_dim, latent_dim, hidden_layers, num_classes):
		""" Initialise Encoder component of CVAE
		Arguments:
		------------------------
			.. input_dim (int) - input channel of data X, the length of flattened image vector
			.. latent_dim (int) - length of latent dim vector in bottle neck layer
			.. hidden_layers (List[int]) - denotes length dimension of hidden layers' weight. 
			.. num_classes (int) - number of classes/target groups in dataset
		"""
		super().__init__()

		if not isinstance(hidden_layers, (list, tuple)) or len(hidden_layers) < 1:
			raise ValueError('Error: CEncoder is initialised without hidden layers.')

		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.num_classes = num_classes

		# Initialise first hidden layer
		self.hidden_layers = nn.ModuleList([
			nn.Linear(input_dim + num_classes, hidden_layers[0])
		])

		# Add more hidden layers
		zipped_layers = zip(hidden_layers[:-1], hidden_layers[1:])
		self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in zipped_layers])

		# Bottle neck layers
		self.mu_layer = nn.Linear(hidden_layers[-1], latent_dim)      # shape: [batch_size, latent_dim]
		self.logvar_layer = nn.Linear(hidden_layers[-1], latent_dim)  # shape: [batch_size, latent_dim]


	def forward(self, x):
		# Input x shape: [batch_size, input_dim + num_classes]
		x = x.view(-1, self.input_dim + self.num_classes)

		# Forward flow to hidden layers
		for linear in self.hidden_layers:
			x = F.relu(linear(x))

		# Latent space parametric variables
		z_mu = self.mu_layer(x)
		z_logvar = self.logvar_layer(x)

		return z_mu, z_logvar


class CDecoder(nn.Module):
	""" Decoder module for Conditional VAE network
	"""

	def __init__(self, input_dim, latent_dim, hidden_layers, num_classes):
		""" Initialise Decoder component of CVAE
		Arguments:
		------------------------
			.. input_dim (int) - input channel of data X, the length of flattened image vector
			.. latent_dim (int) - length of latent dim vector in bottle neck layer
			.. hidden_layers (List[int]) - denotes length dimension of hidden layers' weight. 
			.. num_classes (int) - number of classes/target groups in dataset
		"""
		super().__init__()

		if not isinstance(hidden_layers, (list, tuple)) or len(hidden_layers) < 1:
			raise ValueError('Error: CEncoder is initialised without hidden layers.')

		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.num_classes = num_classes

		# Initialise first hidden layer
		self.hidden_layers = nn.ModuleList([
			nn.Linear(latent_dim + num_classes, hidden_layers[0])
		])

		# Add more hidden layers
		zipped_layers = zip(hidden_layers[:-1], hidden_layers[1:])
		self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in zipped_layers])

		# Recontruction layer
		self.recontruction_layer = nn.Linear(hidden_layers[-1], input_dim)

	def forward(self, z):
		# Latent variable z shape: [batch_size, latent_dim + num_classes]
		x = z.view(-1, self.latent_dim + self.num_classes)

		# Forward flow to hidden layers
		for linear in self.hidden_layers:
			x = F.relu(linear(x))

		# Recontructed input image
		generated_x = F.sigmoid(self.recontruction_layer(x))

		return generated_x


class CVAE(nn.Module):
	""" Conditional VAE (Variational Autoencoder)
	"""

	def __init__(self, input_dim, latent_dim, num_classes):
		""" Initialise Decoder component of CVAE
		Arguments:
		------------------------
			.. input_dim (int) - input channel of data X, the length of flattened image vector
			.. latent_dim (int) - length of latent dim vector in bottle neck layer
			.. hidden_layers (List[int]) - denotes length dimension of hidden layers' weight. 
			.. num_classes (int) - number of classes/target groups in dataset
		"""
		super().__init__()
		
		encoder_hidden_layer = [512, 128, 128]
		decoder_hidden_layer = [128, 128, 512]

		self.encoder = CEncoder(input_dim, latent_dim, encoder_hidden_layer, num_classes)
		self.decoder = CDecoder(input_dim, latent_dim, decoder_hidden_layer, num_classes)

	def forward(self, x, y):
		# Concatenate image x and label y
		x = torch.cat((x, y), dim=1)

		# Learn latent Z distribution parameter
		z_mu, z_logvar = self.encoder(x)

		# Resampling using reparameterisation trick
		std = torch.exp(z_logvar / 2)
		eps = torch.randn_like(std)
		sampled_z = eps * std + z_mu
		
		z = torch.cat((sampled_z, y), dim=1)

		# Recontruct image x
		generated_x = self.decoder(z)

		return generated_x, z_mu, z_logvar

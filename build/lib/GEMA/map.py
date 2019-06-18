import json
import numpy as np
from tqdm.auto import tqdm
from numba import jit

class Map:
	
	def __init__(self, 
			  data,
			  size = -1, 
			  period = 10,
			  initial_lr =0.1,
			  initial_neighbourhood = 0,
			  distance = 'euclidean',
			  use_decay = False,
			  reinforcement=0, 
			  extension=1, 
			  compression=0.5, 
			  normalization='none', 
			  snapshot=0, 
			  presentation='random', 
			  weights='random'):

		# Checking input parameters
		if (period < 1):
			print('Period must be a positive value')
			return -1

		if (initial_lr > 1 or initial_lr < 0):
			print('Learning rate initial must be between 0 and 1')
			return -1

		if (snapshot < 0 and snapshot > period):
			print('Snapshot must be between 0 and period')
			return -1

		if(size != -1):
			# The number of weights should be more than 1 because we have to calculate the second bmu
			if (size < 2 and size != -1):
				print('Map size must be a value higher than 1')
				return -1
			
		self.map_size = size
		self.presentation = presentation
		self.initial_lr = initial_lr
		self.distance = distance
		self.use_decay = use_decay
		self.num_data = data.shape[0]
		self.input_data_dimension = data.shape[1]
		self.period = period
		self.neighbourhood = initial_neighbourhood if initial_neighbourhood is not 0 else size
		# Normalizating input data
		training_data = self.__normalize(data, method = normalization)
		# Initialize weights
		self.weights = 0
		self.weights = self.__init_weights(method = weights, data = data)
		
		
		print("\n\nTRAINING...\n")
		# Input patterns
		num_pattern = 0
		for numPresentation in tqdm(range(1, self.period + 1)):
			if(presentation == 'sequential'):
				# Select patterns sequentialy
				new_pattern = training_data[numPresentation % self.num_data]
			else:
				# Select patterns randomly
				new_pattern = training_data[np.random.randint(0, self.num_data -1)]

			# Getting the winner neuron
			bmu = self.calculate_bmu(new_pattern)

			# Getting learning rate value and current neighbourhood
			eta = Map.variation_learning_rate(self.initial_lr, numPresentation, self.period)
			v_final = 1 if self.use_decay else 0
			v = Map.variation_neighbourhood(self.neighbourhood, numPresentation, self.period, v_final)
			self.__adjust_weights( v, eta, bmu[1], new_pattern)

		print("\n\nFINISHED.")
		
		origin_initial_lr = initial_lr
		# Reinforcement periods
		for reinforcement_number in range(reinforcement):

			# Increasing period and compress learning rate(initial)
			self.period = int(self.period * extension)
			self.reinforcement_lr = origin_initial_lr * compression
			origin_initial_lr = self.initial_lr

			# Training again
			for numPresentation in tqdm(range(1, self.period + 1)):
				if(presentation == 'sequential'):
					# Select patterns sequentialy
					new_pattern = training_data[numPresentation % num_data]
				else:
					# Select patterns randomly
					new_pattern = training_data[np.random.randint(0, self.num_data -1)]

				# Getting the winner neuron
				bmu = self.calculate_bmu(new_pattern)

				# Getting learning rate value and current neighbourhood
				eta = Map.variation_learning_rate(self.initial_lr, numPresentation, self.period)
				v = 1
				self.__adjust_weights( v, eta, bmu[1], new_pattern)

	# GETTING BMU AND SECOND BMU
	def calculate_bmu(self, pattern):
		bmu_pos = np.array([-1, -1])
		bmu = np.array([-1, -1, -1])
		second_bmu_pos = np.array([-1, -1])
		second_bmu = np.array([-1, -1, -1])
		distTemp = -1

		for x in range(0, self.weights.shape[0]):
			for y in range(0, self.weights.shape[1]):

				# Calculating the best, when a new one is found, saving the old one (second best bmu)
				dist = Map.vector_distance(vector1 = pattern,vector2= self.weights[x,y])
				if distTemp == -1 or dist < distTemp:
					distTemp = dist

					# Saving the previous neuron (second best)
					second_bmu = bmu
					second_bmu_pos = bmu_pos

					# Saving the new best
					bmu = self.weights[x][y]
					bmu_pos = np.array([x, y])

		return (bmu, bmu_pos, second_bmu, second_bmu_pos)

	def calculate_distance(self, vector1, vector2):
		if self.distance == 'chebyshev':
			return Map.chebyshev_distance(vector1, vector2)
		elif self.distance == 'euclidean':
			return Map.vector_distance(vector1, vector2)

	# VARIATION OF LEARNING RATE
	@jit(nopython = True)
	def variation_learning_rate(initial_lr, i, iterations_number):
		return initial_lr + ((-initial_lr * i)/ iterations_number)

	# VARIATION NEIGHBOURHOOD
	@jit(nopython = True)
	def variation_neighbourhood(initial_neighbourhood, i, iterations_number, final = 0):
		return final + initial_neighbourhood * (1 - (i / iterations_number))

	# GETTING DECAY VALUE
	@jit(nopython = True)
	def decay(distance_BMU, current_neighbourhood):
		return np.exp(-(distance_BMU**2) / (2* (current_neighbourhood**2)))

	# GETTING DISTANCE BETWEEN VECTORS
	@jit(nopython = True)
	def vector_distance(vector1, vector2):
		result = 0
		for i in range(0, vector1.shape[0]) :
			result = result + (vector1[i] - vector2[i])**2
		return np.sqrt(result)

	# GETTING CHEBYSHEV DISTANCE
	@jit(nopython = True)
	def chebyshev_distance(vector1, vector2):
		return np.max(np.abs(vector1 - vector2))

	# Function to update weights
	def __adjust_weights(self, v, eta, bmu, pattern):
		# Adjusting weights
		for x in range(0, self.map_size):
			for y in range(0, self.map_size):
				if self.calculate_distance(np.array([x, y]), bmu) <= v:
					a = Map.decay(Map.vector_distance(np.array([x, y]), bmu), v) if self.use_decay else 1
					self.weights[x][y] = self.weights[x][y] + eta * a * (pattern - self.weights[x][y])


	
	def __normalize(self, data, method):
		if method is not 'none':
			if method == 'fwn':
				# Feature Wise Normalization

				training_data_temp = data
				mean = training_data_temp.mean(axis=0)
				training_data_temp = training_data_temp - mean
				std = training_data_temp.std(axis=0)
				training_data_temp /= std
				data = training_data_temp

			if method == 'euclidean':
				# Euclidean Normalization
				training_data_temp = data
				data = np.empty(training_data_temp.shape, dtype=float)
				for vector in range (training_data_temp.shape[0] - 1):
					temp = 0
					for component in range (training_data_temp.shape[1] - 1):
						temp = temp + training_data_temp[vector, component] ** 2
					temp = np.sqrt(temp)

					for component in range (training_data_temp.shape[1] - 1):
						data[vector, component] = training_data_temp[vector, component] / temp

		return data

	def __init_weights(self, method, data):
		# Initializing weights if don't exist. If exist, we don't overrite them
		if self.weights == 0 or not 'none':
			if(method == 'random'):
				# Getting the weights from random values between 0 and 1
				return np.random.random(self.input_data_dimension * (self.map_size ** 2)).reshape((self.map_size,self. map_size, self.input_data_dimension))

			elif(method == 'random_negative'):
				# Getting the weights from random values between -1 and 1
				return np.random.uniform(-1, 1, self.input_data_dimension * (self.map_size ** 2)).reshape((self.map_size,self. map_size, self.input_data_dimension))

			elif(method == 'sample'):
				# Getting the weights from the input data (training data)
				total_weights = self.input_data_dimension * (self.map_size ** 2)
				weights_list = []

				for i in range(total_weights):
					weights_list.append(data[np.randint(0, self.num_data - 1)][np.randint(0, self.input_data_dimension - 1)])

				return np.array(weights_list).reshape((self.map_size,self. map_size, self.input_data_dimension))
		else:
			# Checking if the weights has the correct shape
			if((self.weights.shape[0] != self.map_size) or (self.weights.shape[1] != self.map_size) or (self.weights.shape[2] != self.input_data_dimension)):
				return np.array([0])
	
	
	######################################################
	#                    JSON METHODS                    #
	######################################################

	# LOAD CLASSIFIER FROM THE FILE
	def load_classifier(self, filename='Model', verbose = 1):

		# Opening the JSON file and getting all the models
		with open(filename + '.json') as json_file:
			data = json.load(json_file)

			# Reading and setting all the attributes
			for model in data['model']:
				self.map_size = model['map_size']
				self.input_data_dimension = model['input_data_dimension']
				self.weights = np.array(model['weights'])

		# Showing a message to the user
		print('Imported successfully')

	# SAVE CLASSIFIER IN THE FILE
	def save_classifier(self, filename='Model'):
		# Creating the JSON object
		data = {}

		# Setting array
		data['model'] = []

		# Appending the model
		data['model'].append({
			'map_size' : self.map_size,
			'input_data_dimension' : self.input_data_dimension,
			'weights' : self.weights.tolist(),
		})

		# Writing in the file
		with open(filename + '.json', 'w') as outfile:
			json.dump(data, outfile)

		# Showing a message to the user
		print('Saved successfully')
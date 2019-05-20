######################################################
#                    LIBRARIES                       #
######################################################

import numpy as np, math, os, imageio, shutil, pdb
import matplotlib.gridspec as gridspec
import plotly.graph_objs as go, plotly
import plotly.figure_factory as ff
import json
import random as rnd
import pandas as pd
import warnings

from numba import jit

from matplotlib import pyplot as plt
from matplotlib import patches as patches
from sklearn import datasets
from scipy import misc
from IPython.display import display, HTML

from plotly.offline import init_notebook_mode, iplot, plot

class GEMA:

    ######################################################
    #                    ATTRIBUTES                      #
    ######################################################

    # GENERAL PURPOSE KOHONEN ATTRIBUTES
    map_size = 0

    # TRAINING KOHONEN ATTRIBUTES
    period = 0
    initial_lr = 0.0
    end_lr = 0
    neighbourhood = 0
    input_data_dimension = 0
    weights = 0
    num_data = 0
    classification_labels = None
    presentation = 'random'
    initial_weights = 'random'

    # MAPS ATTRIBUTES
    characteristics_data_labels = None
    activations_map = None
    distances_map = None
    classification_map = None

    # GENERAL STATISTICS
    num_activations = 0
    mean_distance_map = 0
    neurons_per_num_activations = None
    topological_map = 0

    ######################################################
    #                    CONSTRUCTOR                     #
    ######################################################

    # The default map_size is -1. All the Kohonen objects that are started from JSON hasn't to give an initial map size.
    def __init__(self, map_size = -1, verbose = 1):
        if(map_size != -1):
            # The number of neurons should be more than 1 because we have to calculate the second bmu
            if (map_size < 2 and map_size != -1):
                print('Map size must be a value higher than 1')

            else:
                self.map_size = map_size
                self.neighbourhood = self.map_size

                if (verbose == 1):
                    print('Object created successfully')

    ######################################################
    #                 TRAINING METHOD                    #
    ######################################################
    def train(self, training_data, period, initial_lr, reinforcement=0, extension=1, compression=0.5, normalization='none', snapshot=0, presentation='random', weights='random', verbose = 1):

        # Checking input parameters
        if (period < 1):
            print('Period must be a positive value')

        if (initial_lr > 1 or initial_lr < 0):
            print('Learning rate initial must be between 0 and 1')

        if (snapshot < 0 and snapshot > period):
            print('Snapshot must be between 0 and period')

        # Initializing parameters
        self.presentation = presentation
        self.training_data = training_data
        self.initial_lr = initial_lr
        origin_initial_lr = initial_lr
        self.num_data = self.training_data.shape[0]
        self.input_data_dimension = training_data.shape[1]
        self.period = period
        bmu = np.array([])
        self.initial_weights = weights

        # Normalizating input data
        if normalization is not 'none':
            if normalization == 'fwn':
                # Feature Wise Normalization
                if verbose == 1:
                    print("Input data normalizated")

                training_data_temp = training_data
                mean = training_data_temp.mean(axis=0)
                training_data_temp = training_data_temp - mean
                std = training_data_temp.std(axis=0)
                training_data_temp /= std
                training_data = training_data_temp

            if normalization == 'euclidean':
                # Euclidean Normalization
                if verbose == 1:
                    print("Input data normalizated")

                training_data_temp = training_data
                training_data = np.empty(training_data_temp.shape, dtype=float)
                for vector in range (training_data_temp.shape[0] - 1):
                    temp = 0
                    for component in range (training_data_temp.shape[1] - 1):
                        temp = temp + training_data_temp[vector, component] ** 2
                    temp = math.sqrt(temp)

                    for component in range (training_data_temp.shape[1] - 1):
                        training_data[vector, component] = training_data_temp[vector, component] / temp

        # Initializing weights if don't exist. If exist, we don't overrite them
        if self.weights == 0 or not 'none':
            if(weights == 'random'):
                # Getting the weights from random values between 0 and 1
                self.weights = np.random.random(self.input_data_dimension * (self.map_size ** 2)).reshape((self.map_size,self. map_size, self.input_data_dimension))

            elif(weights == 'random_negative'):
                # Getting the weights from random values between -1 and 1
                self.weights = np.random.uniform(-1, 1, self.input_data_dimension * (self.map_size ** 2)).reshape((self.map_size,self. map_size, self.input_data_dimension))

            elif(weights == 'sample'):
                # Getting the weights from the input data (training data)
                total_weights = self.input_data_dimension * (self.map_size ** 2)
                weights_list = []

                for i in range(total_weights):
                    weights_list.append(training_data[rnd.randint(0, self.num_data - 1)][rnd.randint(0, self.input_data_dimension - 1)])

                self.weights = np.array(weights_list).reshape((self.map_size,self. map_size, self.input_data_dimension))

            if verbose == 1:
                print('\nNew weights created')

        else:
            # Checking if the weights has the correct shape
            if((self.weights.shape[0] != self.map_size) or (self.weights.shape[1] != self.map_size) or (self.weights.shape[2] != self.input_data_dimension)):
                print('Shape doesn\'t match with input data')

        # Show information to the user
        print("\n\nTRAINING...\n")

        if verbose == 1:
            print("0%... Iteration: 0")

        # Input patterns
        num_pattern = 0
        for numPresentation in range(1, self.period + 1):

            # Show information to the user each 5% iterations
            five = int(self.period / 20)
            for i in range(0, 20):
                if numPresentation == (i * five) and verbose == 1 :
                    print(str(i * 5) + "%... Iteration: " + str(numPresentation))

            # If the user has selected sequential we present patterns sequentially
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if(presentation == 'sequential'):
                new_pattern = self.training_data[num_pattern]

                # Point to the next
                num_pattern += 1

                # If the next is the last one, point to the first of the training data
                if(num_pattern == self.num_data):
                    num_pattern = 0
            else:
                # Choosing a random data if the user choose random
                new_pattern = self.training_data[np.random.randint(0, self.num_data -1)]

            # Getting the winner neuron
            bmu = self.calculate_bmu(new_pattern)

            # Getting learning rate value and current neighbourhood
            eta = self.variation_learning_rate(self.initial_lr, numPresentation, self.period)
            v = self.variation_neighbourhood(self.neighbourhood, numPresentation, self.period)

            # Adjusting weights
            for x in range(0, self.weights.shape[0]):
                for y in range(0, self.weights.shape[1]):
                    if self.vector_distance(np.array([x, y]), bmu[1]) <= self.neighbourhood:
                        self.weights[x][y] = self.weights[x][y] + eta * self.decay(self.vector_distance(np.array([x, y]), bmu[1]), v) * (new_pattern - self.weights[x][y])

        if verbose == 1:
            print("100%... Iteration: " + str(self.period))

        # Showing information to the user in green colour
        print("\n\nFINISHED.")

        # Reinforcement periods
        for reinforcement_number in range(reinforcement):

            # Increasing period and compress learning rate(initial)
            self.period = int(self.period * extension)
            self.initial_lr = origin_initial_lr * compression
            origin_initial_lr = self.initial_lr

            # Show information to the user
            print("\n\nTraining reinforcement " + str(reinforcement_number + 1) + "...\n")
            if(verbose == 1):
                print("New period = " + str(self.period))
                print("New learning rate = " + str(self.initial_lr) + "\n\n")

            # Showing information of the current status
            if verbose == 1:
                print("0%... Iteration: 0")

            # Training again
            for numPresentation in range(1, self.period + 1):

                # Show information to the user each 5% iterations
                five = int(self.period / 20)
                for i in range(0, 20):
                    if numPresentation == (i * five) and verbose == 1:
                        print(
                            str(i * 5) + "%... Iteration: " + str(numPresentation))

                # If the user has selected sequential we present patterns sequentially
                warnings.simplefilter(action='ignore', category=FutureWarning)
                if(presentation == 'sequential'):
                    new_pattern = self.training_data[num_pattern]

                    # Point to the next
                    num_pattern += 1

                    # If the next is the last one, point to the first of the training data
                    if(num_pattern == self.num_data):
                        num_pattern = 0
                else:
                    # Choosing a random data if the user choose random
                    new_pattern = self.training_data[np.random.randint(0, self.num_data -1)]

                # Getting the winner neuron
                bmu = self.calculate_bmu(new_pattern)

                # Getting learning rate value and current neighbourhood
                eta = self.variation_learning_rate(self.initial_lr, numPresentation, self.period)
                v = 1

                # Adjusting weights
                for x in range(0, self.weights.shape[0]):
                    for y in range(0, self.weights.shape[1]):
                        if self.vector_distance(np.array([x, y]), bmu[1]) <= self.neighbourhood:
                            self.weights[x][y] = self.weights[x][y] + eta * self.decay(self.vector_distance(np.array([x, y]), bmu[1]), v) * (new_pattern - self.weights[x][y])


            if verbose == 1:
                print("100%... Iteration: " + str(self.period))

            # Show information to the user
            print("\n\nFinished reinforcement " + str(reinforcement_number + 1) + ".")


    ######################################################
    #              CLASSIFICATION METHOD                 #
    ######################################################
    def classify(self, classification_data, other=None, tagged=False, verbose = 1):

        print("Classificacion data")
        print(classification_data)
        print("Other")
        print(other)
        
        pd.options.mode.chained_assignment = None  # default='warn'

        # If the input data is tagged, keep all the tags; if not, create them
        if(tagged):
            self.classification_labels = classification_data[:, 0]
            self.classification_data = classification_data[:, 1:]
        else:
            self.classification_data = classification_data
            self.classification_labels = np.arange(classification_data.shape[0])

        if(verbose==2):
            print("\n\nTags: \n" + str(self.classification_labels))
            print("\n\nClassification data: \n" + str(self.classification_data))

        # Declaration and initialization
        self.activations_map = np.zeros((self.map_size, self.map_size), dtype=int)
        self.distances_map = np.zeros((self.map_size, self.map_size), dtype=float)
        num = 0
        self.topological_map = 0

        structure = {
                'labels': self.classification_labels.tolist(),
                'data': self.classification_data.tolist(),
                'x': np.zeros(classification_data.shape[0], dtype=int).tolist(),
                'y': np.zeros(classification_data.shape[0], dtype=int).tolist(),
                'dist': np.zeros(classification_data.shape[0], dtype=float).tolist()
            }

        self.classification_map = pd.DataFrame(structure)

        if other is not None:
            self.classification_map = pd.concat([self.classification_map, other], axis=1)

        # Input all the patterns
        for pattern in range(0, classification_data.shape[0]):
            num = num + 1

            # Getting the BMU neuron
            bmu = self.calculate_bmu(classification_data[pattern])

            # Getting the position
            bmu_2DCoordinates = bmu[1]
            bmu_x = bmu_2DCoordinates[0]
            bmu_y = bmu_2DCoordinates[1]

            # If the second best neuron is inside the
            if self.vector_distance(bmu[3], bmu[1]) > 1:
                self.topological_map += 1

            # Saving information in the maps
            distance = self.vector_distance(classification_data[pattern], self.weights[bmu_x][bmu_y])
            self.activations_map[bmu_x][bmu_y] += 1
            self.distances_map[bmu_x][bmu_y] += distance

            self.classification_map['x'][pattern] = bmu_x
            self.classification_map['y'][pattern] = bmu_y
            self.classification_map['dist'][pattern] = distance

        # Number of neurons that have identified pattern
        self.num_activations = np.count_nonzero(self.activations_map != 0)

        # If the divider is 0, setting -1
        self.activations_map[self.activations_map == 0] = -1

        # Calculating the mean distance for each neuron
        self.distances_map = np.absolute(self.distances_map / self.activations_map)

        # Reverting the changes (-1 --> 0)
        self.activations_map[self.activations_map == -1] = 0

        # Calculating the mean of the distances in all the map (only for neurons that have identified a pattern)
        self.mean_distance_map = np.sum(self.distances_map) / self.num_activations

        # Decreasing the number of decimal places to 5
        self.distances_map = np.around(self.distances_map, decimals = 5)

        # Showing information to the user (statistics)
        if verbose == 1:
            self.show_classification_data_summary()


    ######################################################
    #                   SUMMARY METHOD                   #
    ######################################################
    def show_classification_data_summary(self):
        print("CLASSIFICATION DATA REPORT")
        print("\n---- Network Architecture ----")
        print("Map side: " + str(self.map_size))
        print("Input data dimension: " + str(self.input_data_dimension))

        print("\n---- Input data ----")
        print("Number of samples: " + str(self.num_data))
        print("Presentation of samples: " + str(self.presentation))

        print("\n---- Training params ----")
        print("Initial Learning Rate: " + str(self.initial_lr))
        print("Final Learning Rate: " + str(self.end_lr))
        print("Period: " + str(self.period))
        print("Initial Neighborhood: " + str(self.neighbourhood))
        print("Initial weights: " + self.initial_weights)

        print("\n---- Classification results ----")
        print("Topological Map: " + str(self.topological_map))
        print("\nNumber of classes/activations: " + str(self.num_activations))
        print("Activations Map: \n" + str(self.activations_map))
        print("\nMean distance of the map: " + str(self.mean_distance_map))
        print("Distance map: \n" + str(self.distances_map))
        print("\nClassification table:")
        display(self.classification_map)

    ######################################################
    #                   MAPS METHODS                     #
    ######################################################

    # GET CLASSIFICATION TABLE
    def get_classification_table(self):
        return (self.classification_map)

    # GET INFORMATION ABOUT A NEURON
    def info_neuron(self, x, y):
        selection = self.classification_map.loc[
            (self.classification_map['x'] == x)
            & (self.classification_map['y'] == y)]
        return(selection)

    # GET INFORMATION ABOUT ONE PATTERN
    def info_pattern(self, pattern_label):
        selection = self.classification_map.loc[(self.classification_map['labels'] == pattern_label)]
        return(selection)

    # SHOW CLASSIFICATION MAP REPORT
    def classification_map_report(self):
        for i in range(self.map_size):
            for j in range(self.map_size):
                if(self.info_neuron(i,j).index.size > 0):
                    print("\n\nNeurona [" + str(i) + ", " + str(j) + "]\n")
                    display(self.info_neuron(i, j))

    # SET PLOTLY CREDENTIALS
    def set_plotly_credentials(self, username, api_key):
        plotly.tools.set_credentials_file(username=username, api_key=api_key)

    # HEAT MAP
    def heat_map(self, filename='heat_map'):

        # MODIFIED. Activation map rotated 90º so it matches with the Heat Map visualisation
        map_rot = np.rot90(self.activations_map)

        fig = ff.create_annotated_heatmap(map_rot)

        iplot(fig, filename=filename)

    # ELEVATION MAP
    def elevation_map(self, filename = 'elevation_map'):

        # MODIFIED. Activation map rotated 90º so it matches with the Elevation Map visualisation
        map_rot = np.rot90(self.activations_map)

        data = [
            go.Surface(
                z=self.reverse_matrix(map_rot)
            )
        ]
        layout = go.Layout(
            title=filename,
            autosize=False,
            width=1000,
            height=1000,
            margin=dict(
                l=65,
                r=50,
                b=65,
                t=90
            )
        )
        fig = go.Figure(data=data, layout=layout)
        iplot(fig, filename=filename)

    # CHARACTERISTICS GRAPH
    def characteristics_graph(self, row, column, labels=np.array([]), size_x=10, size_y=10, angle=45):
        self.characteristics_data_labels = labels

        data = np.array(self.weights[row][column])
        plt.figure(figsize=(size_x,size_y))
        if(self.characteristics_data_labels.size > 0):
            plt.xticks(np.arange(self.input_data_dimension), self.characteristics_data_labels, rotation=angle)
        display(plt.plot(data, label='[' + str(row) + ',' + str(column) + ']'))

    # BAR CHAR
    def bar_chart(self, data, filename='bar_chart'):
        data_np = np.asarray(data).reshape(-1)
        data_bar = [go.Bar(y=data_np)]
        layout = {
            'xaxis': {'title': 'Times Activated'},
            'yaxis': {'title': 'Number of Neurons'},
            'barmode': 'relative'
            };
        iplot({'data': data_bar, 'layout': layout}, filename=filename)

    # NEURONS PER NUM ACTIVATIONS
    def neurons_per_num_activations_map(self, filename='neurons_per_num_activations_map', save=False):
        num_max_activations = np.max(self.activations_map) + 1
        neurons_per_num_activations = np.zeros(num_max_activations)

        for i in range(0, num_max_activations):
            neurons_per_num_activations[i] = np.count_nonzero(self.activations_map == i)

        self.bar_chart(data=neurons_per_num_activations, filename=filename)


    ######################################################
    #                     CORE METHODS                   #
    ######################################################
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
                dist = self.vector_distance(pattern, self.weights[x][y])
                if distTemp == -1 or dist < distTemp:
                    distTemp = dist

                    # Saving the previous neuron (second best)
                    second_bmu = bmu
                    second_bmu_pos = bmu_pos

                    # Saving the new best
                    bmu = self.weights[x][y]
                    bmu_pos = np.array([x, y])

        return (bmu, bmu_pos, second_bmu, second_bmu_pos)

    # VARIATION OF LEARNING RATE
    @staticmethod
    @jit(nopython = True)
    def variation_learning_rate(initial_lr, i, iterations_number):
        return initial_lr + ((-initial_lr * i)/ iterations_number)

    # VARIATION NEIGHBOURHOOD
    @staticmethod
    @jit(nopython = True)
    def variation_neighbourhood(initial_neighbourhood, i, iterations_number):
        return 1 + initial_neighbourhood * (1 - (i / iterations_number))

    # GETTING DECAY VALUE
    @staticmethod
    @jit(nopython = True)
    def decay(distance_BMU, current_neighbourhood):
        return np.exp(-(distance_BMU**2) / (2* (current_neighbourhood**2)))

    # GETTING DISTANCE BETWEEN VECTORS
    @staticmethod
    @jit(nopython = True)
    def vector_distance(vector1, vector2):
        result = 0
        for i in range(0, vector1.shape[0]) :
            result = result + (vector1[i] - vector2[i])**2
        return math.sqrt(result)

    # GETTING REVERSE MATRIX
    @staticmethod
    @jit(nopython = True)
    def reverse_matrix(matrix):
        new_matrix_weights = np.copy(matrix)
        for i in range(0, new_matrix_weights.shape[0]):
            new_matrix_weights[i] = matrix[matrix.shape[0] - 1 - i]
        return new_matrix_weights


    ######################################################
    #                GETTERS AND SETTERS                 #
    ######################################################

    def get_map_size(self):
        return self.map_size

    def get_period(self):
        return self.period

    def get_initial_lr(self):
        return self.initial_lr

    def get_end_lr(self):
        return self.end_lr

    def get_neighbourhood(self):
        return self.neighbourhood

    def get_training_data(self):
        return self.training_data

    def get_input_data_dimension(self):
        return self.input_data_dimension

    def get_weights(self):
        return self.weights

    def get_num_data(self):
        return self.num_data

    def get_characteristics_data_labels(self):
        return self.characteristics_data_labels

    def get_activations_map(self):
        return self.activations_map

    def get_distances_map(self):
        return self.distances_map

    def get_num_activations(self):
        return self.num_activations

    def get_mean_distance_map(self):
        return self.mean_distance_map

    def get_neurons_per_num_activations(self):
        return self.neurons_per_num_activations

    def set_map_size(self, value):
        self.map_size = value

    def set_period(self, value):
        self.period = value

    def set_initial_lr(self, value):
        self.initial_lr = value

    def set_end_lr(self, value):
        self.end_lr = value

    def set_neighbourhood(self, value):
        self.neighbourhood = value

    def set_training_data(self, value):
        self.training_data = value

    def set_input_data_dimension(self, value):
        self.input_data_dimension = value

    def set_weights(self, value):
        self.weights = value

    def set_num_data(self, value):
        self.num_data = value

    def set_characteristics_data_labels(self, value):
        self.characteristics_data_labels = value

    def set_activations_map(self, value):
        self.activations_map = value

    def set_distances_map(self, value):
        self.distances_map = value

    def set_num_activations(self, value):
        self.num_activations = value

    def set_mean_distance_map(self, value):
        self.mean_distance_map = value

    def set_neurons_per_num_activations(self, value):
        self.neurons_per_num_activations = value


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


    # LOAD TRAINER FROM THE FILE
    def load_trainer(self, filename='Session', verbose = 1):

        # Opening the JSON file and getting all the models
        with open(filename + '.json') as json_file:
            data = json.load(json_file)

            # Reading and setting all the attributes
            for model in data['model']:
                self.map_size = model['map_size']
                self.period = model['period']
                self.initial_lr = model['initial_lr']
                self.end_lr = model['end_lr']
                self.neighbourhood = model['neighbourhood']
                self.input_data_dimension = model['input_data_dimension']
                self.weights = np.array(model['weights'])
                self.num_data = model['num_data']
                self.topological_map = model['topological_map']
                self.activations_map = np.array(model['activations_map'])
                self.distances_map = np.array(model['distances_map'])
                self.num_activations = model['num_activations']
                self.mean_distance_map = np.array(model['mean_distance_map'])
                self.neurons_per_num_activations = model['neurons_per_num_activations']
                self.classification_labels = np.array(model['classification_labels'])
                self.presentation = model['presentation']
                self.initial_weights = model['initial_weights']

        self.classification_map = pd.read_json(filename + '_cmdf.json')

        # Showing a message to the user
        print('Imported successfully')

    # SAVE TRAINER IN THE FILE
    def save_trainer(self, filename='Session'):
        # Creating the JSON object
        data = {}

        # Setting array
        data['model'] = []

        # Appending the model
        data['model'].append({
            'map_size' : self.map_size,
            'period' : self.period,
            'initial_lr' : self.initial_lr,
            'end_lr' : self.end_lr,
            'neighbourhood' : self.neighbourhood,
            'input_data_dimension' : self.input_data_dimension,
            'weights' : self.weights.tolist(),
            'num_data' : self.num_data,
            'topological_map' : self.topological_map,
            'activations_map' : self.activations_map.tolist(),
            'distances_map' : self.distances_map.tolist(),
            'num_activations' : self.num_activations,
            'mean_distance_map' : self.mean_distance_map.tolist(),
            'neurons_per_num_activations' : self.neurons_per_num_activations,
            'classification_labels' : self.classification_labels.tolist(),
            'presentation' : self.presentation,
            'initial_weights' : self.initial_weights,
        })

        # Writing in the file
        with open(filename + '.json', 'w') as outfile:
            json.dump(data, outfile)

        # Saving in other file the classification map table
        self.classification_map.to_json(filename + '_cmdf.json')

        # Showing a message to the user
        print('Saved successfully')


    # SHOW DATA READ
    def show_model(self):
        print('Map size: \n' + str(self.map_size) + '\n\n')
        print('Period: \n' + str(self.period) + '\n\n')
        print('Initial learning rate: \n' + str(self.initial_lr) + '\n\n')
        print('End learning rate: \n' + str(self.end_lr) + '\n\n')
        print('Neighbourhood: \n' + str(self.neighbourhood) + '\n\n')
        print('Input data dimension: \n' + str(self.input_data_dimension) + '\n\n')
        print('Weights: \n' + str(self.weights) + '\n\n')
        print('Number of data: \n' + str(self.num_data) + '\n\n')
        print('Activations map: \n' + str(self.activations_map) + '\n\n')
        print('Distances map: \n' + str(self.distances_map) + '\n\n')
        print('Total number of activations: \n' + str(self.num_activations) + '\n\n')
        print('Mean distance map: \n' + str(self.mean_distance_map) + '\n\n')

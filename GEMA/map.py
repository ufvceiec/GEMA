import json
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

class Map:
    """
    Map class is the main component of GEMA. It contains the classifying map that allows for classification and
    is subject of analysis in search of data information
    """

    def __init__(self,
                 data=None,
                 size=-1,
                 period=10,
                 initial_lr=0.1,
                 initial_neighbourhood=0,
                 distance='euclidean',
                 use_decay=False,
                 normalization='none',
                 presentation='random',
                 weights='random',
                 topology='rectangular'):
        """Initializing the map requires some information provided

        :param data: numpy array of 2 dimensions. First dimension corresponds to data samples, while the second
        represents an specific sample's data
        :param size: size of the side of the map. Resulting map will be of dimension: size x size x data_depth
        :param period: Number of iterations to train the map for. A small number will produced an undertrained
        map, while a bigger one will compress the borders, resulting in many activations on the sides of the map
        :param initial_lr: Learning rate determines how much neurons will move on the map
        :param initial_neighbourhood: Initial neighbourhood determines how many neurons will learn. If none is
        provided, it will default to size
        :param distance: Determines the way distance will be calculated. Current options include:
            - 'euclidean'
            - 'chebyshev'
        :param use_decay: If set to True, neurons further away from the bmu will learn less than those closer
        :param normalization: Allows some normalization of input data. We do not recommend doing this, but normalizing
        the data previously. Current options include:
            - 'none': Perform no normalization to data
            - 'fwn': Normalizes each feature independently
            - '01scale': Scales data to 0-1 interval so it set data closer to weights
            - 'euclidean': Euclidean Normalization
        :param presentation: If set to 'sequential' data will be presented sequentially, otherwise it will be presented
        randomly
        :param weights: Technique used to initialize the weights. Current options include:
            - 'random': From 0 to 1
            - 'random_negative': From -1 to 1
            - 'sample': Takes samples from data. This is useful if data is not normalized
            - 'PCA': sequence of vectors taken along a hyperplane spanned by the two largest principal components of the dataset.
        :param topology: Grid topology to use. Options:
            - 'rectangular': standard square grid (default)
            - 'hexagonal': offset hexagonal grid — produces rounder, more uniform neighbourhoods
        """

        # Checking input parameters
        self.__trainned = False
        assert period > 1, 'Period must be a positive value'

        assert initial_lr < 1 or initial_lr > 0, 'Learning rate initial must ' \
                                                 'be between 0 and 1'

        assert size >= 2, 'Map size must be a value higher than 1'

        assert topology in ('rectangular', 'hexagonal'), \
            "topology must be 'rectangular' or 'hexagonal'"

        self.map_size = size
        self.presentation = presentation
        self.initial_lr = initial_lr
        self.distance = distance
        self.use_decay = use_decay
        self.num_data = 0
        self.input_data_dimension = 0
        self.period = period
        self.neighbourhood = initial_neighbourhood if initial_neighbourhood != 0 \
            else size
        self.normalization = normalization
        self.presentation = presentation
        self.weights_init = weights
        self.topology = topology

        # Initialize weights
        self.weights = np.random.random(1)

        # Create neuron position matrix used for neighbourhood calculations.
        # For rectangular topology each neuron sits at integer grid coordinates [y, x].
        # For hexagonal topology every other row is offset by 0.5 in x and all rows
        # are spaced sqrt(3)/2 apart in y, giving a proper hex lattice.
        self.__ids_matrix = self.__build_ids_matrix()

        if data is not None:
            self.train(data)

    # ------------------------------------------------------------------
    # GRID HELPERS
    # ------------------------------------------------------------------

    def __build_ids_matrix(self):
        """Build the (map_size, map_size, 2) array of neuron 2-D positions."""
        if self.topology == 'rectangular':
            ids = []
            for y in range(self.map_size):
                row = []
                for x in range(self.map_size):
                    row.append([float(y), float(x)])
                ids.append(row)
            return np.array(ids)          # shape (size, size, 2)

        else:  # hexagonal
            # Offset-coordinate hex grid.
            # Odd rows (0-indexed) are shifted right by 0.5.
            # Vertical spacing = sqrt(3)/2 ≈ 0.866 so that
            # all six nearest neighbours are equidistant.
            sqrt3_2 = np.sqrt(3) / 2
            ids = []
            for y in range(self.map_size):
                row = []
                x_offset = 0.5 if (y % 2 == 1) else 0.0
                for x in range(self.map_size):
                    row.append([y * sqrt3_2, x + x_offset])
                ids.append(row)
            return np.array(ids)          # shape (size, size, 2)

    def train(self,
              data):
        """ Trains the map

        :param data: numpy array of 2 dimensions. First dimension corresponds to data samples, while the second
        represents an specific sample's data
        :return:
        """
        self.num_data = data.shape[0]
        self.input_data_dimension = data.shape[1]
        # Normalizating input data
        training_data = self.__normalize(data, method=self.normalization)
        self.weights = self.__init_weights(data=data, method=self.weights_init)

        print("TRAINING...")
        # Input patterns
        for numPresentation in tqdm(range(1, self.period + 1)):
            if self.presentation == 'sequential':
                # Select patterns sequentialy
                new_pattern = training_data[numPresentation % self.num_data]
            else:
                # Select patterns randomly
                new_pattern = training_data[np.random.randint(0, self.num_data)]

            # Getting the winner neuron
            bmu = self.calculate_bmu(new_pattern)

            # Getting learning rate value and current neighbourhood
            eta = self.variation_learning_rate(self.initial_lr, numPresentation,
                                               self.period)
            v_final = 1 if self.use_decay else 0
            v = self.variation_neighbourhood(self.neighbourhood, numPresentation,
                                             self.period, v_final)
            self.__adjust_weights(v, eta, bmu[1], new_pattern)

        self.__trainned = True
        print("FINISHED.")

    def reinforce(self, training_data,
                  reinforcement=0,
                  extension=1,
                  compression=0.5, ):
        """Reinforce learning

        :param training_data: numpy array of 2 dimensions. First dimension corresponds to data samples, while the second
        represents an specific sample's data
        :param reinforcement: Number of reinforcement iterations
        :param extension: Multiplier applied to the current period each reinforcement round (>1 extends training).
        :param compression: Multiplier applied to the learning rate each reinforcement round (<1 shrinks it).
        :return:
        """
        origin_initial_lr = self.initial_lr
        # Reinforcement periods
        for reinforcement_number in range(reinforcement):

            # Increasing period and compress learning rate(initial)
            self.period = int(self.period * extension)
            reinforcement_lr = origin_initial_lr * compression
            origin_initial_lr = reinforcement_lr
            self.initial_lr = reinforcement_lr

            # Training again
            for numPresentation in tqdm(range(1, self.period + 1)):
                if self.presentation == 'sequential':
                    # Select patterns sequentialy
                    new_pattern = training_data[numPresentation % self.num_data]
                else:
                    # Select patterns randomly
                    new_pattern = training_data[np.random.randint(0, self.num_data)]

                # Getting the winner neuron
                bmu = self.calculate_bmu(new_pattern)

                # Getting learning rate value and current neighbourhood
                eta = self.variation_learning_rate(self.initial_lr, numPresentation,
                                                   self.period)
                v = 1
                self.__adjust_weights(v, eta, bmu[1], new_pattern)

    # GETTING BMU AND SECOND BMU
    def calculate_bmu(self, pattern):
        """Calculates Best Machine Unit (BMU) for a concrete pattern.
        The bmu is the neuron whose weight vector is closest in the space to the pattern.
        It also calculates the second closest neuron, used for metrics calculation on the map
        :param pattern: array of the pattern we are comparing
        :return:
            - bmu: distance of the bmu
            - bmu_pos: coordinates of the bmu
            - second_bmu: distance of the second bmu
            - second_bmu_pos: coordinates of the second bmu
        """
        distancias = np.sum((self.weights - pattern) ** 2, axis=-1)

        bmu = np.min(distancias, axis=None)
        bmu_pos = np.unravel_index(np.argmin(distancias, axis=None), distancias.shape)

        distancias[bmu_pos] = np.inf

        second_bmu = np.min(distancias, axis=None)
        second_bmu_pos = np.unravel_index(np.argmin(distancias, axis=None), distancias.shape)

        return bmu, bmu_pos, second_bmu, second_bmu_pos

    def calculate_distance(self, vector1, vector2):
        """Function that selects which distance will be calculated between the two input vectors
        :param vector1: First vector
        :param vector2: Second vector
        :return: distance between vector1 and vector2
        """
        if self.distance == 'chebyshev':
            return Map.chebyshev_distance(vector1, vector2)
        elif self.distance == 'euclidean':
            return Map.vector_distance(vector1, vector2)

    # VARIATION OF LEARNING RATE
    @staticmethod
    def variation_learning_rate(initial_lr, i, iterations_number):
        """Funtion to calculate the learning rate for the current iteration relative from the total number of iterations

        :param initial_lr: initial learning rate
        :param i: Current iteration
        :param iterations_number: total number of iterations
        :return: learning rate for the current iteration
        """
        return initial_lr + ((-initial_lr * i) / iterations_number)

    # VARIATION NEIGHBOURHOOD
    @staticmethod
    def variation_neighbourhood(initial_neighbourhood, i,
                                iterations_number, final=0):
        """Function to calculate the neighbourhood variation for the current iteration

        :param initial_neighbourhood: Initial neighbourhood
        :param i: Current iteration
        :param iterations_number: total number of iterations
        :param final: neighbourhood for the last iteration
        :return: neighbourhood for the current iteration
        """
        return final + initial_neighbourhood * (1 - (i / iterations_number))

    # GETTING DECAY VALUE
    @staticmethod
    def decay(distance_BMU, current_neighbourhood):
        """Function that calculates decay from the bmu. The closer a neuron is to the bmu, the lower the decay will be

        :param distance_BMU: Array of the distances to the bmu
        :param current_neighbourhood: current neighbourhood
        :return: decay array
        """
        return np.exp(-(distance_BMU ** 2) / (2 * (current_neighbourhood ** 2)))

    # GETTING DISTANCE BETWEEN VECTORS
    @staticmethod
    def vector_distance(vector1, vector2):
        """ Function that calculates the euclidean distance between vector1 and vector2

        :param vector1:
        :param vector2:
        :return:
        """
        return np.sqrt(np.sum(np.square(vector1 - vector2), axis=-1))

    # GETTING CHEBYSHEV DISTANCE
    @staticmethod
    def chebyshev_distance(vector1, vector2):
        """ Function that calculates the chebyshev distance between vector1 and vector2

        :param vector1:
        :param vector2:
        :return:
        """
        return np.max(np.abs(vector1 - vector2))

    # Function to update weights
    def __adjust_weights(self, v, eta, bmu, pattern):
        """ Function that adjust our map's weigths considering v, eta, bmu and pattern

        :param v: Neighbourhood variation for current iteration
        :param eta: Learning rate variation for current iteration
        :param bmu: BMU coordinates from where to calculate distance's matrix
        :param pattern: current iteration pattern that we are checking
        :return:
        """
        # For hexagonal topology the bmu index must be converted to its
        # actual 2-D position in the hex lattice before computing distances.
        bmu_pos_2d = self.__ids_matrix[bmu[0], bmu[1]]
        distances = self.calculate_distance(self.__ids_matrix, bmu_pos_2d)
        to_update = distances <= v

        if self.use_decay:
            decay = self.decay(distances, v)
        else:
            decay = np.ones_like(distances)

        self.weights[to_update] = self.weights[to_update] + \
                                  eta * np.expand_dims(decay[to_update], axis=1) * \
                                  (pattern - self.weights[to_update])

    @staticmethod
    def __normalize(data, method):
        """Function that normalizes data

        :param data: data to normalize
        :param method: method to use. Available options are:
            - 'none': Perform no normalization to data
            - 'fwn': Normalizes each feature independently
            - '01scale': Scales data to 0-1 interval so it set data closer to weights
            - 'euclidean': Euclidean (L2) normalization — each sample vector is divided
              by its L2 norm so that all samples lie on the unit hypersphere.
        :return:
        """
        if method != 'none':
            if method == 'fwn':
                # Feature Wise Normalization
                training_data_temp = data
                mean = training_data_temp.mean(axis=0)
                training_data_temp = training_data_temp - mean
                std = training_data_temp.std(axis=0)
                training_data_temp /= std
                data = training_data_temp

            if method == 'euclidean':
                # Vectorised L2 normalisation: divide every row by its own norm.
                # Rows with zero norm are left as-is to avoid division by zero.
                norms = np.linalg.norm(data, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                data = data / norms

            if method == '01scale':
                data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def __init_weights(self, data, method):
        """ Function to initialize the weights matrix

        :param data: Data used to train the map
        :param method: Method to initialize the map. Available options include:
            - 'random': From 0 to 1
            - 'random_negative': From -1 to 1
            - 'sample': Takes samples from data. This is useful if data is not normalized
            - 'PCA': sequence of vectors taken along a hyperplane spanned by the two largest principal components of the dataset.
        :return:
        """
        if method == 'random':
            # Getting the weights from random values between 0 and 1
            return np.random.random(self.input_data_dimension *
                                    (self.map_size ** 2)).reshape(
                (self.map_size, self.map_size, self.input_data_dimension))

        elif method == 'random_negative':
            # Getting the weights from random values between -1 and 1
            return np.random.uniform(-1, 1, self.input_data_dimension *
                                     (self.map_size ** 2)).reshape(
                (self.map_size, self.map_size, self.input_data_dimension))

        elif method == 'sample':
            # Getting the weights from the input data (training data)
            idx = np.random.randint(0, self.num_data,
                                    size=(self.map_size, self.map_size))
            return data[idx]   # shape (size, size, input_data_dimension)

        elif method == 'PCA':
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(data)
            pca_weights = np.zeros((self.map_size, self.map_size, self.input_data_dimension))
            pc_length, pc = np.linalg.eig(np.cov(np.transpose(standardized_data)))
            pc_order = np.argsort(-pc_length)
            for i, c1 in enumerate(np.linspace(-1, 1, self.map_size)):
                for j, c2 in enumerate(np.linspace(-1, 1, self.map_size)):
                    pca_weights[i, j] = c1 * pc[:, pc_order[0]] + c2 * pc[:, pc_order[1]]
            return pca_weights

    ######################################################
    #                    JSON METHODS                    #
    ######################################################

    # LOAD CLASSIFIER FROM THE FILE
    @classmethod
    def load_classifier(cls, filename='Model'):

        # Opening the JSON file and getting all the models
        with open(filename + '.json') as json_file:
            data = json.load(json_file)

            # Reading and setting all the attributes
            for model in data['model']:
                map_size = model['map_size']
                input_data_dimension = model['input_data_dimension']
                presentation = model['presentation']
                initial_lr = model['initial_lr']
                distance = model['distance']
                use_decay = model['use_decay']
                num_data = model['num_data']
                period = model['period']
                neighbourhood = model['neighbourhood']
                topology = model.get('topology', 'rectangular')   # backwards-compatible
                weights = np.array(model['weights'])

        new_map = Map(data=None,
                      size=map_size,
                      period=period,
                      initial_lr=initial_lr,
                      initial_neighbourhood=neighbourhood,
                      distance=distance,
                      use_decay=use_decay,
                      topology=topology,
                      )

        new_map.weights = weights
        new_map.input_data_dimension = input_data_dimension
        new_map.presentation = presentation
        new_map.num_data = num_data
        new_map.__trainned = True
        # Showing a message to the user
        print('Imported successfully')

        return new_map

    # SAVE CLASSIFIER IN THE FILE
    def save_classifier(self, filename='Model'):
        # Creating the JSON object
        data = {'model': []}

        # Appending the model
        data['model'].append({
            'map_size': self.map_size,
            'input_data_dimension': self.input_data_dimension,
            'presentation': self.presentation,
            'initial_lr': self.initial_lr,
            'distance': self.distance,
            'use_decay': self.use_decay,
            'num_data': self.num_data,
            'period': self.period,
            'neighbourhood': self.neighbourhood,
            'topology': self.topology,
            'weights': self.weights.tolist()
        })

        # Writing in the file
        with open(filename + '.json', 'w') as outfile:
            json.dump(data, outfile)

        # Showing a message to the user
        print('Saved successfully')

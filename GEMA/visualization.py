import imageio
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
from IPython.display import HTML, display
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from matplotlib import cm
from plotly.offline import init_notebook_mode, iplot, plot 



class visualization:
	# SET PLOTLY CREDENTIALS
	@staticmethod
	def set_plotly_credentials(username, api_key):
		plotly.tools.set_credentials_file(username=username, api_key=api_key)

	# HEAT MAP
	@staticmethod
	def heat_map(classification, filename='heat_map', colorscale = 'Reds', cmax = 0):
		init_notebook_mode(connected=True)
		# MODIFIED. Activation map rotated 90º so it matches with the Heat Map visualisation
		map_rot = np.transpose(np.rot90(classification.activations_map))
		cmax = np.max(classification.activations_map) if cmax is 0 else cmax
		fig = ff.create_annotated_heatmap(map_rot, showscale= True, colorscale = colorscale, zmin=0, zmax = cmax)

		iplot(fig, filename=filename)

	# ELEVATION MAP
	@staticmethod
	def elevation_map(classification, filename = 'elevation_map'):
		init_notebook_mode(connected=True)

		# MODIFIED. Activation map rotated 90º so it matches with the Elevation Map visualisation
		map_rot = np.rot90(classification.activations_map)

		data = [
			go.Surface(
				z=np.fliplr(map_rot),
				colorscale='Reds'
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
	@staticmethod
	def characteristics_graph(map, row, column, labels=np.array([]), size_x=10, size_y=10, angle=45):
		map.characteristics_data_labels = labels

		data = np.array(map.weights[row][column])
		plt.figure(figsize=(size_x,size_y))
		if(map.characteristics_data_labels.size > 0):
			plt.xticks(np.arange(map.input_data_dimension), map.characteristics_data_labels, rotation=angle)
		display(plt.plot(data, label='[' + str(row) + ',' + str(column) + ']'))


	# CHARACTERISTICS GRAPH
	@staticmethod
	def characteristics_bargraph(map, row, column, labels=np.array([]), size_x=10, size_y=10, angle=45):
		map.characteristics_data_labels = labels

		data = np.array(map.weights[row][column])
		plt.figure(figsize=(size_x,size_y))
		if(map.characteristics_data_labels.size > 0):
			plt.xticks(np.arange(map.input_data_dimension), map.characteristics_data_labels, rotation=angle)
		rainbow = cm.get_cmap('tab20',data.shape[0])
		display(plt.bar(np.arange(data.shape[0]),data, label='[' + str(row) + ',' + str(column) + ']', color = rainbow(np.linspace(0,1,data.shape[0]))))

	# BAR CHAR
	@staticmethod
	def bar_chart(data, filename='bar_chart'):
		init_notebook_mode(connected=True)
		data_np = np.asarray(data).reshape(-1) 
		data_bar = [go.Bar(y=data_np)]
		layout = {
			'xaxis': {'title': 'Times Activated'},
			'yaxis': {'title': 'Number of Neurons'},
			'barmode': 'relative'
			}
		iplot({'data': data_bar, 'layout': layout}, filename=filename)

	# NEURONS PER NUM ACTIVATIONS
	@staticmethod
	def neurons_per_num_activations_map(classification, filename='neurons_per_num_activations_map', save=False):
		num_max_activations = np.max(classification.activations_map) + 1
		neurons_per_num_activations = np.zeros(num_max_activations)

		for i in range(0, num_max_activations):
			neurons_per_num_activations[i] = np.count_nonzero(classification.activations_map == i)

		visualization.bar_chart(data=neurons_per_num_activations, filename=filename)


	#TODO Comprobar el funcionamiento correcto
	# Print a codebook vector specified by index
	@staticmethod
	def codebook_vector(map, index = 0, header = 'none', filename='codebook_vector'):
		init_notebook_mode(connected=True)

		map_rot = np.transpose(np.rot90(np.around(map.weights[:,:,index], decimals = 2)))

		fig = ff.create_annotated_heatmap(map_rot, showscale= True)
		if(header is not 'none'):
			fig.layout.title = header

		# Make text size smaller
		for i in range(len(fig.layout.annotations)):
			fig.layout.annotations[i].font.size = 7

		iplot(fig, filename=filename)

	# Print all codebooks for a som
	@staticmethod
	def codebook_vectors(map, headers = np.array([])):
		if(headers.size < 1):
			headers = np.arange(map.input_data_dimension) 
		for i in range(0, map.input_data_dimension):
			visualization.codebook_vector(map, i, str(headers[i]))

	@staticmethod
	def umatrix(classification, colorscale = 'binary'):
		plt.imshow(np.rot90(classification.umatriz), cmap=colorscale)
		plt.colorbar()
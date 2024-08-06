# Import libraries
#  Import library for loading data file such as .xlxs or text file
from numpy import loadtxt

# Sequential class used to create a linear stack of layers of simple neural network
from keras.models import Sequential

#Dense layers are fully connected layers, where each neuron  connects to all neuron in previous layer
from keras.layers import Dense

dataset= loadtxt("Data_set.xlsx")


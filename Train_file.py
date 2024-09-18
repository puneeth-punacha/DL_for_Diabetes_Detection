# Import libraries
#  Import library for loading data file such as .xlsx or text file
from numpy import loadtxt

# Sequential class used to create a linear stack of layers of simple neural network
from keras.models import Sequential

#Dense layers are fully connected layers, where each neuron  connects to all neuron in previous layer
from keras.layers import Dense
import numpy as np
dataset= loadtxt('Data_set.csv',delimiter=',')
n=np.shape(dataset)[0]-100
x= dataset[:n,0:8] # All data set

print(np.shape(x))
y= dataset[:n,8] # Class variable
print(y)

# Creating a neural network

model= Sequential()

#Layer 1
model.add(Dense(8, input_dim= 8, activation= 'relu'))
#Layer 2
#model.add(Dense(10, activation= 'relu'))
#Layer 2
#model.add(Dense(10, activation= 'relu'))
#Layer 2
model.add(Dense(8, activation= 'relu'))
#Layer 3
model.add(Dense(1, activation= 'sigmoid'))

# compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'] )
model.fit(x,y,epochs=50, batch_size=30)

a,b= model.evaluate(x,y)
print("accuracy %.2f" % (b*100))
# Write the model's structure

model_json= model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

# Write model's weights
model.save_weights("model.weights.h5")
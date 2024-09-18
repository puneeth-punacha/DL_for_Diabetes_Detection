from numpy import loadtxt
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
import numpy as np

dataset= loadtxt('Data_set.csv', delimiter=',')
n=np.shape(dataset)[0]
x= dataset[101:n,0:8]
y= dataset[101:n, 8]

json_file= open('model.json','r')
loaded_model_from_json= json_file.read()
model= model_from_json(loaded_model_from_json)
model.load_weights("model.weights.h5")
print("Model loaded from disk")

predictions= model.predict(x)

print("test accuracy %.2f" % ((np.sqrt(mean_squared_error(y, predictions))*100)))

for i in range(5,10):
    print('%s => %d (expected %d)' %(x[i].tolist(), predictions[i],y[i]))


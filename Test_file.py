from numpy import loadtxt
from keras.models import model_from_json

dataset= loadtxt('Data_set.csv', delimiter=',')
x= dataset[:,0:8]
y= dataset[:, 8]

json_file= open('model.json','r')
loaded_model_from_json= json_file.read()
model= model_from_json(loaded_model_from_json)
model.load_weights("model.weights.h5")
print("Model loaded from disk")

predictions= model.predict(x)

for i in range(5,10):
    print('%s => %d (expected %d)' %(x[i].tolist(), predictions[i],y[i]))


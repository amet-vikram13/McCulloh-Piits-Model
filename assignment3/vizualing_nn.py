# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz
import numpy
import pandas as pd
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = pd.read_csv("heart.csv").values
# split into input (X) and output (Y) variables
X = dataset[:,:13]
Y = dataset[:,13]
# create model
model = Sequential()
model.add(Dense(45, input_dim=13, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=100,batch_size=303)
model.summary()
# evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#ann_viz(model, title="Neural Network Schematic")

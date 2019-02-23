# IMPORTING LIBRARY
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# IMPORTING FILE AND READING DATASET


def read_data():
    f = pd.read_csv("DATA.csv")
    print('No. of Rows (or) Variables : ', len(f.index))
    print('No. of Coulumns (or) samples : ', len(f.columns))
    x = f.loc[1:45031]
    # print('X: ', x)
    X = x.T
    print('Shape of input data:', x.shape)
    y = f.tail(1)
    # print('label :', y)
    Y = y.T
    print('Shape of label data:', y.shape)

# Encode THE DEPENDENT VARIABLES

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    Z = one_hot_encode(Y)
    return X, Z

# ENCODING FUNCTION


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# READ THE DATASET


X, Z = read_data()

# SHUFFLE THE DATASET
X, Z = shuffle(X, Z, random_state=1)

# CONVERT THE DATASET INTO TRAINING AND TESTING DATASET
train_X, test_X, train_y, test_y = train_test_split(X, Z, test_size=0.20, random_state=45030)

# DEFINING THE IMPORTANT PARAMETERS AND VARIABLES
learning_rate = 0.001
training_epochs = 100
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
n_class = 2

# DEFINING No. OF HIDDEN LAYERS AND No. OF NEURONS
n_hidden_1 = 45
n_hidden_2 = 45
n_hidden_3 = 45
n_hidden_4 = 45

x = tf.placeholder(tf.float32, [None, n_dim])
w = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


# DEFINING THE MODEL
def multilayer_perceptron(x, weights, biases):

    # HIDDEN LAYERS ARE ACTIVATED BY RELU
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

# ACTIVATION OF OUTPUT LAYER BY LINEARITY

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# DEFINING WEIGHT AND BIASES OF EACH LAYER
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# INITIALIZING THE VARIABLE
init = tf.global_variables_initializer()

# CALLING THE MODEL
y = multilayer_perceptron(x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)


# CALCULATE THE COST AND ACCURACY
mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_X, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_X, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict={x: test_X})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: test_X, y_: test_y}))
    accuracy_history.append(accuracy)

    print('epoch : ', epoch, ' - ', 'cost : ', cost, " - MSE : ", mse_, " - Accuracy : ", accuracy)


# PLOT MSE AND ACCURACY GRAPH
# MSE GRAPH
plt.plot(mse_history, 'r')
plt.xlabel('Epoch Number')
plt.ylabel('Mean Square error')
plt.show()
# ACCURACY GRAPH
plt.plot(accuracy_history)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.show()
# Cost graph
plt.plot(cost_history)
plt.xlabel('Epoch Number')
plt.ylabel('Loss Function')
plt.show()

# PRINT THE FINAL ACCURACY
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("accuracy: ", (sess.run(accuracy, feed_dict={x: test_X, y_: test_y})))

# PRINT THE FINAL MSE
pred_y = sess.run(y, feed_dict={x: test_X})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE : %.4f" % sess.run(mse))

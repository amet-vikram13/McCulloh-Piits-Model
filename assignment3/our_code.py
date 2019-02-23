import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def initialize_parameters(layers_dims) :
    parameters = {}
    L = len(layers_dims)-1

    for l in range(1,L+1) :
        parameters["W"+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))

    return parameters

def sigmoid(Z) :
    A = 1/(1+np.exp(-Z))
    A = np.clip(A,0.0000001,0.9999999)
    return A

def relu(Z) :
    A = np.maximum(Z,0)
    return A

def linear_forward(A_prev,W,b) :
    Z = np.dot(W,A_prev)+b
    cache = (A_prev,b,W,Z)
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation) :
    Z, cache = linear_forward(A_prev,W,b)
    A = 0
    if activation == "sigmoid" :
        A = sigmoid(Z)
    elif activation == "relu" :
        A = relu(Z)
    cache = (cache[0],cache[1],cache[2],cache[3],A)
    return A,cache  

def L_model_forward(X, parameters) :
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1,L) :
        A_prev = A
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        A,cache = linear_activation_forward(A_prev,W,b,"relu")
        caches.append(cache)
    
    W = parameters["W"+str(L)]
    b = parameters["b"+str(L)]
    AL, cache = linear_activation_forward(A,W,b,"sigmoid")
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y) :
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL))) / m
    cost = np.squeeze(cost)
    return cost

def sigmoid_backward(dA,Z) :
    s = sigmoid(Z)
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA,Z) :
    dZ = np.array(dA,copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_backward(dZ,cache) :
    A_prev, b, W = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T) / m 
    db = np.sum(dZ,axis=1,keepdims=True) / m
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

def linear_activation_backward(dA_next,cache,activation) :
    A_prev,b,W,Z,A = cache
    if activation=="sigmoid" :
        dZ = sigmoid_backward(dA_next,Z)
        dA_prev,dW,db = linear_backward(dZ,(A_prev,b,W))
    
    elif activation=="relu" :
        dZ = relu_backward(dA_next,Z)
        dA_prev,dW,db = linear_backward(dZ,(A_prev,b,W))

    return dA_prev,dW,db

def L_model_backward(AL,Y,caches) :
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))

    curr_cache = caches[L-1]
    dA_prev,dW,db = linear_activation_backward(dAL,curr_cache,"sigmoid")
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db
    
    for l in range(L-2,-1,-1) :
        curr_cache = caches[l]
        dA_prev,dW,db = linear_activation_backward(dA_prev,curr_cache,"relu")
        grads["dW"+str(l+1)] = dW
        grads["db"+str(l+1)] = db

    return grads

def update_parameters(parameters,grads,learning_rate) :
    L = len(parameters) // 2

    for l in range(L) :
        parameters["W"+str(l+1)] -= grads["dW"+str(l+1)]*learning_rate
        parameters["b"+str(l+1)] -= grads["db"+str(l+1)]*learning_rate
    
    return parameters

def prediction(X,parameters) :

    L = len(parameters) // 2
    A_prev = X

    for l in range(1,L) :
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        Z = np.dot(W,A_prev)+b
        A_prev = relu(Z)
    
    W = parameters["W"+str(L)]
    b = parameters["b"+str(L)]
    Z = np.dot(W,A_prev)+b
    y = sigmoid(Z)
    y[y <= 0.5] = 0
    y[y > 0.5]  = 1
    return y

def accuracy(y_pred,y_test) :
    m = y_pred.shape[1]
    corr = 0
    for i in range(m) :
        if y_pred[:,i]==y_test[:,i] :
            corr += 1
    accu = (corr / m)*100
    return accu

def train_test_split(X,y,split) :
    indices = np.random.permutation(X.shape[1])
    split = int(split*X.shape[1])
    train_idx, test_idx = indices[:split],indices[split:]
    X_train, X_test = X[:,train_idx], X[:,test_idx]
    y_train, y_test = y[:,train_idx], y[:,test_idx]
    return X_train,y_train,X_test,y_test

def prepare_minibatch(X,y,minibatch_size) :
    m = X.shape[1]
    indices = np.random.permutation(m)
    X,y = X[:,indices],y[:,indices]
    
    batches = []
    num_minibatch = minibatch_size // m

    X_batch = X[:,:minibatch_size*num_minibatch+1]
    y_batch = y[:,:minibatch_size*num_minibatch+1]
    X_rem = X[:,minibatch_size*num_minibatch+1:]
    y_rem = y[:,minibatch_size*num_minibatch+1:]

    for i in range(0,len(X_batch)-1,minibatch_size) :
        batches.append((X_batch[:,i:i+minibatch_size],y_batch[:,i:i+minibatch_size]))

    batches.append((X_rem,y_rem))

    return batches

def model(X,y,layer_dims,split=0.8,epochs=10,learning_rate=0.001,minibatch=False) :
    X_train,y_train,X_test,y_test = train_test_split(X,y,split)
    
    if minibatch :
        minibatch_size = int(input("Enter minibatch size : "))
    
    parameters = initialize_parameters(layer_dims)
    
    epoch_cost = []

    for epoch in range(epochs) :
        print("######## EPOCH {} ########".format(epoch))
        cost = 0
        batches = [(X_train,y_train)]
        if minibatch :
            batches = prepare_minibatch(X_train,y_train,minibatch_size)
        for batch in batches :
            X_batch, y_batch = batch
            AL,caches = L_model_forward(X_batch,parameters)
            cost += compute_cost(AL,y_batch)
            grads = L_model_backward(AL,y_batch,caches)
            parameters = update_parameters(parameters,grads,learning_rate)
        print("Cost after epoch {} : {}\n\n".format(epoch,cost))
        epoch_cost.append(cost)
    
    y_pred = prediction(X_test,parameters)
    accu = accuracy(y_pred,y_test)
    return accu,epoch_cost

def main() :
    df = pd.read_csv("heart.csv").values
    
    ### Preprocessing - No scaling of data ###
    idx = np.random.permutation(df.shape[0])
    df = df[idx]
    y  = df[:,13]
    y  = y.reshape(1,-1)
    X  = df[:,:13].T
    
    ### Running the Model ###
    layer_dims = [13,45,45,45,45,1]
    accu,cost = model(X,y,layer_dims,0.8,100,0.001,False)
    
    print("ACCURACY : {}".format(accu))
    plt.plot(cost,'r')
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.show()
    

if __name__=="__main__":
    main()









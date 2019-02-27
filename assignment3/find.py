import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class DeepNeuralNetwork(object) :
    def __init__(self,dimensions) :
        self.layer_dims = dimensions
    
    def process_data(self,df) :
        minn = df["age"].min()
        maxx = df["age"].max()
        df["age"] = (df["age"] - minn) / (maxx - minn)
        minn = df["trestbps"].min()
        maxx = df["trestbps"].max()
        df["trestbps"] = (df["trestbps"] - minn) / (maxx - minn)
        minn = df["chol"].min()
        maxx = df["chol"].max()
        df["chol"] = (df["chol"] - minn) / (maxx - minn)
        minn = df["thalach"].min()
        maxx = df["thalach"].max()
        df["thalach"] = (df["thalach"] - minn) / (maxx - minn)
        df = df.values
        idx = np.random.permutation(df.shape[0])
        df = df[idx]
        self.y  = df[:,13]
        self.y  = self.y.reshape(1,-1)
        self.X  = df[:,:13].T
    
    def train_test_split(self,split=0.8) :
        indices = np.random.permutation(self.X.shape[1])
        split = int(split*self.X.shape[1])
        train_idx, test_idx = indices[:split],indices[split:]
        self.X_train, self.X_test = self.X[:,train_idx], self.X[:,test_idx]
        self.y_train, self.y_test = self.y[:,train_idx], self.y[:,test_idx]
        return self.X_train,self.y_train
    
    def initialize_parameters(self) :
        self.parameters = {}
        L = len(self.layer_dims)-1
        for l in range(1,L+1) :
            self.parameters["W"+str(l)] = np.random.randn(self.layer_dims[l],self.layer_dims[l-1])*np.sqrt(2/self.layer_dims[l-1])
            self.parameters["b"+str(l)] = np.zeros((self.layer_dims[l],1))
        return self.parameters

    def _sigmoid(self,Z) :
        A = 1/(1+np.exp(-Z))
        A = np.clip(A,0.0000001,0.9999999)
        return A

    def _relu(self,Z) :
        A = np.maximum(Z,0)
        return A

    def _linear_forward(self,A_prev,W,b) :
        Z = np.dot(W,A_prev)+b
        cache = (A_prev,b,W,Z)
        return Z,cache

    def _linear_activation_forward(self,A_prev,W,b,activation) :
        Z, cache = self._linear_forward(A_prev,W,b)
        A = 0
        if activation == "sigmoid" :
            A = self._sigmoid(Z)
        elif activation == "relu" :
            A = self._relu(Z)
        cache = (cache[0],cache[1],cache[2],cache[3],A)
        return A,cache  

    def _L_model_forward(self) :
        caches = []
        A = self.X_train
        L = len(self.parameters) // 2

        for l in range(1,L) :
            A_prev = A
            W = self.parameters["W"+str(l)]
            b = self.parameters["b"+str(l)]
            A,cache = self._linear_activation_forward(A_prev,W,b,"relu")
            caches.append(cache)
        
        W = self.parameters["W"+str(L)]
        b = self.parameters["b"+str(L)]
        AL, cache = self._linear_activation_forward(A,W,b,"sigmoid")
        caches.append(cache)
        return AL,caches

    def _compute_cost(self,AL) :
        m = self.y_train.shape[1]
        cost = -np.sum(np.multiply(self.y_train,np.log(AL)) + np.multiply(1-self.y_train,np.log(1-AL))) / m
        cost = np.squeeze(cost)
        return cost

    def _sigmoid_backward(self,dA,Z) :
        s = self._sigmoid(Z)
        dZ = dA * s * (1-s)
        return dZ

    def _relu_backward(self,dA,Z) :
        dZ = np.array(dA,copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _linear_backward(self,dZ,cache) :
        A_prev, b, W = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ,A_prev.T) / m 
        db = np.sum(dZ,axis=1,keepdims=True) / m
        dA_prev = np.dot(W.T,dZ)

        return dA_prev,dW,db

    def _linear_activation_backward(self,dA_next,cache,activation) :
        A_prev,b,W,Z,A = cache
        if activation=="sigmoid" :
            dZ = self._sigmoid_backward(dA_next,Z)
            dA_prev,dW,db = self._linear_backward(dZ,(A_prev,b,W))
        
        elif activation=="relu" :
            dZ = self._relu_backward(dA_next,Z)
            dA_prev,dW,db = self._linear_backward(dZ,(A_prev,b,W))

        return dA_prev,dW,db

    def _L_model_backward(self,AL,caches) :
        grads = {}
        L = len(caches)
        self.y_train = self.y_train.reshape(AL.shape)
        
        dAL = -(np.divide(self.y_train,AL) - np.divide(1-self.y_train,1-AL))

        curr_cache = caches[L-1]
        dA_prev,dW,db = self._linear_activation_backward(dAL,curr_cache,"sigmoid")
        grads["dW"+str(L)] = dW
        grads["db"+str(L)] = db
        
        for l in range(L-2,-1,-1) :
            curr_cache = caches[l]
            dA_prev,dW,db = self._linear_activation_backward(dA_prev,curr_cache,"relu")
            grads["dW"+str(l+1)] = dW
            grads["db"+str(l+1)] = db

        return grads

    def _update_parameters(self,grads,learning_rate) :
        L = len(self.parameters) // 2

        for l in range(L) :
            self.parameters["W"+str(l+1)] -= grads["dW"+str(l+1)]*learning_rate
            self.parameters["b"+str(l+1)] -= grads["db"+str(l+1)]*learning_rate
        
        return self.parameters

    def run(self,epochs=100,learning_rate=0.001,verbose=True) :        
        t = {"accu":0,"mse":0,"plt_cost":0,"plt_mse":0,"plt_accu":0}	
        self.epoch_cost = []
        self.epoch_accu = []
        self.epoch_mse  = []
        for epoch in range(epochs) :
            if verbose and epoch%10==0 :
                print("######## EPOCH {} ########".format(epoch+1))
            cost = 0
            AL,caches = self._L_model_forward()
            cost += self._compute_cost(AL)
            grads = self._L_model_backward(AL,caches)
            self.parameters = self._update_parameters(grads,learning_rate)
            if verbose and epoch%10==0 :
                print("Cost after epoch {} : {}\n\n".format(epoch+1,cost))
            self.epoch_cost.append(cost)
            self.prediction("train")
            self.epoch_mse.append(self.mse_error("train"))
            self.epoch_accu.append(self.accuracy("train"))
        self.prediction("test")
        accu = self.accuracy("test")
        mse  = self.mse_error("test")
        t["accu"] = accu
        t["mse"]  = mse
        t["plt_cost"] = self.epoch_cost
        t["plt_mse"]  = self.epoch_mse
        t["plt_accu"] = self.epoch_accu
        return t
    
    def plot_cost(self) :
        plt.plot(self.epoch_cost,'r')
        plt.xlabel("epoch")
        plt.ylabel("cost")
        plt.show()
    
    def plot_accu(self) :
        plt.plot(self.epoch_accu,'b')
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.show()
    
    def plot_mse(self) :
        plt.plot(self.epoch_mse,'r')
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.show()

    def prediction(self,data="test") :
        L = len(self.parameters) // 2
        if data=="test" :
            A_prev = self.X_test
        elif data=="train" :
            A_prev = self.X_train

        for l in range(1,L) :
            W = self.parameters["W"+str(l)]
            b = self.parameters["b"+str(l)]
            Z = np.dot(W,A_prev)+b
            A_prev = self._relu(Z)
        
        W = self.parameters["W"+str(L)]
        b = self.parameters["b"+str(L)]
        Z = np.dot(W,A_prev)+b
        self.y_pred = self._sigmoid(Z)
        self.y_pred[self.y_pred <= 0.5] = 0
        self.y_pred[self.y_pred > 0.5]  = 1
        return self.y_pred
    
    def mse_error(self,data="test") :
        if data=="test" :
            y_corr = self.y_test
        elif data=="train" :
            y_corr = self.y_train
        m = y_corr.shape[1]
        err = np.sum(np.square(y_corr - self.y_pred)) / m
        return err

    def accuracy(self,data="test") :
        if data=="test" :
            y_corr = self.y_test
        elif data=="train" :
            y_corr = self.y_train
        m = y_corr.shape[1]
        corr = 0
        for i in range(m) :
            if self.y_pred[:,i]==y_corr[:,i] :
                corr += 1
        accu = (corr / m)*100
        return accu









import numpy as np
import matplotlib.pyplot as plt

class AdalineModel(object) :
    def __init__(self,num_inputs=2,examples=4,threshold=0) :
        self.inp = num_inputs
        self.m   = examples
        self.thr = threshold
        self.W = -1
        self.b = -1
        self.X = -1
        self.y = -1
        self.X_test = -1
        self.y_test = -1
        self.y_quantize = -1
        self.Z = -1
        self.A = -1
    
    def initialize_parameters(self) :
        # random sample in range [-1,1)     
        self.W = -1 + 2*np.random.rand(self.inp,1)
        self.b = -1 + 2*np.random.rand(1,self.m)
        return self.W,self.b
    
    def initialize_data(self) :
        # random sample in range [-1,1)
        self.X = -1 + 2*np.random.rand(self.inp,self.m)
        self.y = -1 + 2*np.random.rand(1,self.m)
        self.X_test = -1 + 2*np.random.rand(self.inp,self.m)
        self.y_test = -1 + 2*np.random.rand(1,self.m)
        return self.X,self.y
    
    def give_data(self,X,y) :
        assert (X.shape==(self.inp,self.m))
        assert (y.shape==(1,self.m))
        self.X = X
        self.y = y
        return self.X,self.y
    
    def _net_input(self) :
        self.Z = np.dot(self.W.T,self.X) + self.b
    
    def _activation(self) :
        self.A = self.Z
    
    def _error(self) :
        err = np.sum(np.square(np.absolute(self.y - self.A)))
        err = np.squeeze(err)
        return err

    def _update_parameters(self,learning_rate) :
        delta = (self.y - self.A)*learning_rate
        for i in range(self.m) :
            self.W = self.W + delta[:,i][0]*self.X[:,i].reshape(-1,1)
        self.b = self.b + delta
    
    def run(self,epochs=5,learning_rate=0.01) :
        self.epoch_err = []
        for e in range(epochs) :
            print(" #### EPOCH {} ####".format(e+1))
            self._net_input()
            self._activation()
            err = self._error()
            self.epoch_err.append(err)
            self._update_parameters(learning_rate)
            print("error : {}".format(err))
    
    def quantize(self) :
        self.y_quantize = self.A
        self.y_quantize[self.y_quantize <= self.thr] = -1
        self.y_quantize[self.y_quantize > self.thr]  = 1
        self.y_quantize = self.y_quantize.astype("int")
        return self.y_quantize

    def rmse(self,data="test") :
        if data=="test" :
            y_pred = np.dot(self.W.T,self.X_test) + self.b
            y_corr = self.y_test
        elif data=="train" :
            y_pred = self.A
            y_corr = self.y
        rmse = np.sqrt(np.sum(np.square(y_corr - y_pred)) / self.m)
        rmse = np.squeeze(rmse)
        print("{} RMSE : {}".format(data,rmse))

    def plot_error(self) :
        plt.plot(self.epoch_err,'r')
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()


    


    

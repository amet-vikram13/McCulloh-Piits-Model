import numpy as np
import matplotlib.pyplot as plt

class AdalineModel(object) :
    def __init__(self,num_inputs=2,examples=4,threshold=-1) :
        self.inp = num_inputs
        self.m   = examples
        self.thr = threshold
        self.W = -1
        self.b = -1
        self.X = -1
        self.y = -1
        self.X_test = -1
        self.y_test = -1
        self.Z = -1
        self.A = -1
    
    def initialize_parameters(self) :
        # random sample in range [-1,1)     
        self.W = -1 + 2*np.random.rand(self.inp,1)
        #self.b = -1 + 2*np.random.rand(1,self.m)    ## b shape (1,m)
        self.b = -1 + 2*np.random.rand()             ## b shape (1,1)
        return self.W,self.b
    
    def initialize_data(self) :
        # random sample in range [-1,1)
        self.X = -1 + 2*np.random.rand(self.inp,self.m)
        self.y = -1 + 2*np.random.rand(1,self.m)
        self.X_test = -1 + 2*np.random.rand(self.inp,self.m)
        self.y_test = -1 + 2*np.random.rand(1,self.m)
        return self.X,self.y
    
    def give_data(self,X,y,split=0.8) :
        self.m = int(split*X.shape[1])
        self.X = X[:,:self.m]
        self.y = y[:,:self.m]
        self.X_test = X[:,self.m:]
        self.y_test = y[:,self.m:]
        #self.b = -1 + 2*np.random.rand(1,self.m)   ## b shape (1,m)
        assert (self.X.shape==(self.inp,self.m))
        assert (self.y.shape==(1,self.m))
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
        delta = np.squeeze(np.sum(delta) / self.m)  ## b shape (1,1)
        self.b = self.b + delta
    
    def _quantize(self) :
        self.A[self.A <= self.thr] = -1
        self.A[self.A  > self.thr] = 1
        self.A = self.A.astype("int")
    
    def run(self,epochs=5,learning_rate=0.01) :
        self.epoch_err = []
        for e in range(epochs) :
            print(" #### EPOCH {} ####".format(e+1))
            self._net_input()
            self._activation()
            if self.thr!=-1 :
                self._quantize()
            err = self._error()
            self.epoch_err.append(err)
            self._update_parameters(learning_rate)
            print("error : {}".format(err))
    
    def predict(self) :
        #y_pred = np.dot(self.W.T,self.X_test) + self.b[:self.y_test.shape[1]]  ## b shape (1,m)
        y_pred = np.dot(self.W.T,self.X_test) + self.b                          ## b shape (1,1)
        if self.thr!=-1 :
            y_pred[y_pred <= self.thr] = -1
            y_pred[y_pred  > self.thr] = 1
            y_pred = y_pred.astype("int")
            corr = 0
            for i in range(y_pred.shape[1]) :
                if y_pred[:,i]==self.y_test[:,i] :
                    corr += 1
            accu = (corr / y_pred.shape[1])*100
            print("ACCURACY : {}".format(accu))
        else :
            rmse = np.sqrt(np.sum(np.square(self.y_test - y_pred)) / y_pred.shape[1])
            print("RMSE : {}".format(rmse))
    
    def plot_error(self) :
        plt.plot(self.epoch_err,'r')
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()

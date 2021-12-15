from NeuralNetworks import NeuralNetworks as NN
import numpy as np

training_file = './Data/bank-note/train.csv'
testing_file = './Data/bank-note/test.csv'

layer_1 = np.array([[-1,1],[-2,2],[-3,3]])
layer_2 = np.array([[-1,1],[-2,2],[-3,3]])
layer_3 = np.array([-1,2,-1.5])
weights = [layer_1, layer_2, layer_3]
x = np.array([1,1])

def backprop_exp():
    y = NN.loss_backpropagation(x, 1, weights, True)
    print(y)

if __name__=="__main__":
    backprop_exp()
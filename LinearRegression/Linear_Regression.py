from os import stat
import numpy as np

class Linear_Regression:
    @staticmethod
    def read_data(filename):
        data = np.genfromtxt(filename, delimiter=',')
        outputs = data[:, -1]
        data = data[:, :-1]
        numb_ex = data.shape[0]
        bias_term = np.ones((numb_ex, 1))
        data = np.append(data, bias_term, axis=1)
        return data, outputs

    def __init__(self, training_file, test_file, learning_rate, optimization_func) -> None:
        self.training_examples, self.training_outputs = Linear_Regression.read_data(training_file)
        self.test_examples, self.test_outputs = Linear_Regression.read_data(test_file)
        self.w, self.costs = optimization_func(self.training_examples, self.training_outputs, learning_rate)

    @staticmethod
    def batch_gradient_descent(training_examples, training_ouputs, r):
        X = training_examples
        y = training_ouputs

        numb_feat = X.shape[1]
        w = np.zeros(numb_feat)
        last_w = np.ones(numb_feat)
        costs = []
        # it = 0
        while(np.linalg.norm(w - last_w) > 1e-6):
            # print(it)
            # it+=1
            # print(w)
            cost = Linear_Regression.calculate_cost(X, y, w)
            costs.append(cost)
            last_w = np.copy(w)
            dJ = -X.T @ (y - X @ w)
            w = w - r * dJ
        return w, costs

    @staticmethod
    def stochastic_gradient_descent(training_examples, training_ouputs, r):
        X = training_examples
        y = training_ouputs

        numb_feat = X.shape[1]
        w = np.zeros(numb_feat)
        last_w = np.ones(numb_feat)
        costs = []
        numb_ex = X.shape[0]

        while(np.linalg.norm(w - last_w) > 1e-6):
            cost = Linear_Regression.calculate_cost(X, y, w)
            costs.append(cost)
            last_w = np.copy(w)
            i = np.random.choice(numb_ex)
            Xi = X[i]
            yi = y[i]
            dJ = - Xi * (yi - w @ Xi)
            w = w - r * dJ
        return w, costs

    @staticmethod
    def calculate_cost(X, y, w):
        return 0.5 * np.sum((y - np.sum((X * w), axis=1)) ** 2)
        
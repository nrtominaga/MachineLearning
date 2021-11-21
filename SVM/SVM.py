import numpy as np
import scipy.optimize as opt

def shuffle_data(examples, labels):  # shuffles our data and labels together
    p = np.random.permutation(examples.shape[0]) # get a permutation
    return examples[p], labels[p] # index based on permutation

def open_data(filename):
    examples = np.genfromtxt(filename, delimiter=',') # read in data
    labels = examples[:, -1] # index labels
    labels = 2 * labels - 1 # map {0,1} -> {-1,1}
    examples = np.delete(examples, -1, 1) # delete each label from example
    ones = np.ones(examples.shape[0])[:, np.newaxis]
    examples = np.append(examples, ones, 1) # append a 1 as bias term to each example
    return examples, labels

def calculate_guess(example, weights):
    guess = 0  # what our final guess is
    for feature_index, feature in enumerate(example):  # go through each feature in our example
        weight = weights[feature_index]  # get the weight for our feature
        guess += weight * feature  # add the weight * feature to our guess
    guess = -1 if guess < 0 else 1  # get the sign for our guess and return that
    return guess

def split_ex(examples, labels, k): # split ex into k
    num_ex = np.arange(examples.shape[0])
    splits = np.array_split(num_ex, k) # split into k regions
    folds = []
    for f in range(k):
        val_ex = examples[splits[f]] # get the validation ex
        train_ex = np.delete(examples, splits[f], axis=0) # get the train ex
        val_labels = labels[splits[f]] # do the same with labels
        train_labels = np.delete(labels, splits[f], axis=0)
        folds.append((train_ex, val_ex, train_labels, val_labels)) # this set is a fold
    return folds

class SVM:
    def __init__(self, training_file, testing_file, epochs, gamma, C, a=None, KfoldCV=1):
        self.training_examples, self.training_labels = open_data(training_file) # get our train ex & labels
        self.testing_examples, self.testing_labels = open_data(testing_file) # test ex & labels
        if KfoldCV > 1: # perform k fold cv
            folds = split_ex(self.training_examples, self.training_labels, KfoldCV) # split into our folds
            total_error = 0
            for f in folds: # for each fold
                train_ex, val_ex, train_labels, val_labels = f # unpack needed ex and labels
                weights = SVM.ssgd_svm(train_ex, train_labels, epochs, gamma, C, a) # run our algo
                err = SVM.error(weights, val_ex, val_labels) # get error on val set
                total_error += err
            self.avg_error = total_error / KfoldCV # average error 
        else: # just perform our algorithm
            self.weights = SVM.ssgd_svm(self.training_examples, self.training_labels, epochs, gamma, C, a)

    @staticmethod
    def error(weights, examples, labels):
        guesses = np.sum(weights * examples, axis=1) # our guesses
        guesses[guesses < 0] = -1 # if they are neg -> -1
        guesses[guesses >= 0] = 1 # otherwise -> 1
        return np.sum(guesses != labels)/examples.shape[0] # error

    @staticmethod
    def ssgd_svm(train_ex, train_labels, epochs, gamma_0, C, a=None):
        a = gamma_0 if a == None else a # our type of covergence
        numb_feats = train_ex.shape[1]
        weights = np.zeros(numb_feats) # inititalize weight vector
        N = train_ex.shape[0]
        for t in range(epochs): # 1...T
            train_ex, train_labels = shuffle_data(train_ex, train_labels) # shuffle the data
            gamma = gamma_0 / (1+(gamma_0/a)*t)
            for ex_idx, ex in enumerate(train_ex): # go thru each ex
                guess = -1 if np.sum(ex * weights) < 0 else 1 # get our guess
                y = train_labels[ex_idx] # y
                w_0 = weights[:-1] # get rid of biases for w_0
                if y != guess:
                    weights = weights - gamma * np.append(w_0, 0) + gamma * C * N * y * ex
                else:
                    weights[:-1] = (1 - gamma) * w_0
        return weights

class Dual_SVM:
    def __init__(self, training_file, testing_file, C, gamma=None):
        self.training_data = open_data(training_file) # get our train ex & labels
        self.testing_data = open_data(testing_file) # test ex & labels
        self.gamma = gamma
        self.weights, self.alphas = Dual_SVM.dualsvm(self.training_data, C, gamma) # run dual svm

    def error(self, error_type='testing', data=None):
        if error_type=='training':
            X, y = self.training_data
        elif error_type=='testing':
            X, y = self.testing_data
        else:
            assert data is not None
            X, y = data
        guesses = np.zeros(y.shape)
        if self.gamma != None:
            tex, tlabels = self.training_data
            for x_index, x in enumerate(X):
                guess = 0
                for i, x_i in enumerate(tex):
                    y_i = tlabels[i]
                    alpha_i = self.alphas[i]
                    guess += y_i * alpha_i * np.exp(- (np.linalg.norm(x_i - x) ** 2)/self.gamma)
                guesses[x_index] = -1 if guess < 0 else 1
        else:
            guesses = X @ self.weights
        guesses[guesses < 0] = -1
        guesses[guesses >= 0] = 1
        return np.sum(guesses != y)/X.shape[0] # error

    @staticmethod
    def dualsvm(train_data, C, gamma=None):
        X, y = train_data # unwrap the data
        numb_ex = X.shape[0]
        kernel = None
        if gamma != None: #lets calcualte kernel now since it's not necessary to calculate over and over again
            kernel = Dual_SVM.calculate_gaussian_kernel(X, gamma)
        else:
            kernel = X @ X.T
        yy = y[:, np.newaxis] @ y[np.newaxis, :]
        bounds = [(0, C)] * numb_ex # setup our bounds
        constraints = [{'type':'eq', 'fun':Dual_SVM.contraints, 'args':[y]}] # setup our constraints
        alphas = np.full(numb_ex, C) # init alphas to C

        solution = opt.minimize(Dual_SVM.objective_func, alphas, args=[yy, kernel], method='SLSQP', bounds=bounds, constraints=constraints) # run our optimization
        alphas = solution.x # recover our optimized alphas
        weights = np.sum(X * y[:, np.newaxis] * alphas[:, np.newaxis], axis=0) # calculate weight

        return weights, alphas # return weight, alphas


    @staticmethod
    def objective_func(alphas, args):
        yy = args[0]
        kernel = args[1]
        aa = alphas[:, np.newaxis] @ alphas[np.newaxis, :]
        summation = np.sum(kernel * yy * aa)

        return 1/2 * summation - np.sum(alphas)

    @staticmethod
    def contraints(alphas, *args):
        return alphas @ args[0]

    @staticmethod
    def calculate_gaussian_kernel(X, gamma):
        numb_ex = X.shape[0]
        gauss = np.zeros((numb_ex, numb_ex))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                gauss[i][j] = np.exp(- (np.linalg.norm(x_i - x_j) ** 2)/gamma)
        return gauss

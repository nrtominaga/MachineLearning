import random


def shuffle_data(examples, labels):  # shuffles our data and labels together
    data = list(zip(examples, labels))
    random.shuffle(data)
    examples, labels = zip(*data)
    return list(examples), list(labels)


def open_data(filename):
    examples = []  # represents our set of examples
    labels = []  # our labels
    with open(filename) as data:  # open the file
        for example in data:  # for each example
            example = example.replace('\n', '').split(
                ',')  # replace each end line with an empty string and then split
            example = [float(feature) if '.' in feature else int(feature) for feature in
                       example]  # make a float if it
            # has a . and make a int otherwise
            examples.append(example)  # add to set of examples
            if example[-1] == 0:  # check if label is zero if it is make it -1
                example[-1] = example[-1] - 1
            labels.append(example.pop())  # add the label to our label list and get rid of it from our features
            example.append(1)  # append the "constant" feature
    return examples, labels


def calculate_guess(example, weights):
    guess = 0  # what our final guess is
    for feature_index, feature in enumerate(example):  # go through each feature in our example
        weight = weights[feature_index]  # get the weight for our feature
        guess += weight * feature  # add the weight * feature to our guess
    guess = -1 if guess < 0 else 1  # get the sign for our guess and return that
    return guess


def calculate_new_weights(example, label, weights, learning_rate):
    new_weights = []  # where we store our new calculate weights
    for index, feature in enumerate(example):  # go through each feature in our example
        new_weights.append(weights[index] + learning_rate * label * feature)  # get the our current weight and calculate
        # the new one
    return new_weights


class Perceptron:
    def __init__(self, training_file, testing_file, learning_rate, epoch, perceptron_algo):
        self.training_examples, self.training_labels = open_data(training_file)  # get training examples and labels
        self.testing_examples, self.testing_labels = open_data(testing_file)  # get testing examples and labels
        self.weights, self.error_function = perceptron_algo(self.training_examples, self.training_labels, learning_rate,
                                                            epoch)  # perform our perceptron

    def perform_error_function(self):
        return self.error_function(self.testing_examples, self.testing_labels, self.weights)

    @staticmethod
    def perceptron(training_examples, training_labels, learning_rate, epoch):
        numb_features = len(training_examples[0])  # find how many features we have
        weights = [0] * numb_features  # and initialize the weights with 0
        for i in range(epoch):  # do epoch number of iterations
            training_examples, training_labels = shuffle_data(training_examples, training_labels)  # shuffle our data
            for example_index, example in enumerate(training_examples):  # go through each training example
                y_guess = calculate_guess(example, weights)  # get the guess using our current weights
                y_actual = training_labels[example_index]  # get our training label for this example
                if y_guess != y_actual:  # if they don't equal calculate the new weights
                    weights = calculate_new_weights(example, y_actual, weights, learning_rate)
        return weights, Perceptron.check_std_perceptron_error  # return weights and error function

    @staticmethod
    def voted_perceptron(training_examples, training_labels, learning_rate, epoch):
        numb_features = len(training_examples[0])  # find number of features
        weights = [0] * numb_features  # initialize weight list with zeros
        current_count = 1  # our current count
        weight_vectors = []  # will store our counts and weight vectors
        for i in range(epoch):  # do epoch number of loops
            for example_index, example in enumerate(training_examples):  # go through every example
                y_guess = calculate_guess(example, weights)  # get the guess for our current example and weights
                y_actual = training_labels[example_index]  # get our actual label
                if y_guess != y_actual:  # if the guess is wrong
                    weight_vectors.append((current_count, weights))  # add weight vector and current count
                    weights = calculate_new_weights(example, y_actual, weights, learning_rate)  # our new weights vector
                    current_count = 1  # reset current count
                else:  # if we have a correct guess
                    current_count += 1  # increment the count for the current weight vector
        # TODO: maybe get rid of the first element in the weight vector
        return weight_vectors, Perceptron.check_voted_perceptron_error  # return counts and weight vector

    @staticmethod
    def averaged_perceptron(training_examples, training_labels, learning_rate, epoch):
        numb_features = len(training_examples[0])  # find number of features
        weights = [0] * numb_features  # initialize weight list with zeros
        averages = [0] * numb_features  # a calculation of our averages
        for i in range(epoch):  # to epoch iterations
            for example_index, example in enumerate(training_examples):  # go through each example
                y_guess = calculate_guess(example, weights)  # get our guess
                y_actual = training_labels[example_index]  # get our label
                if y_guess != y_actual:  # if the guess is incorrect
                    weights = calculate_new_weights(example, y_actual, weights, learning_rate)  # calculate new weights
                averages = [a + w for a, w in zip(averages, weights)]  # get the new averages list
        return averages, Perceptron.check_avg_perceptron_error  # return our averages

    @staticmethod
    def check_std_perceptron_error(testing_examples, labels, weights):
        error_count = 0  # a count for how many examples we get wrong
        numb_examples = len(testing_examples)  # how many examples we have
        for example_index, example in enumerate(testing_examples):
            y_guess = calculate_guess(example, weights)  # get our guess with our weights
            y_actual = labels[example_index]  # get the actual labels
            if y_guess != y_actual:  # if they don't match up increase the error count
                error_count += 1
        return error_count / numb_examples  # calculate error

    @staticmethod
    def check_voted_perceptron_error(testing_examples, labels, weights_vector):
        error_count = 0  # a count for how many examples we get wrong
        numb_examples = len(testing_examples)  # how many examples we have
        for example_index, example in enumerate(testing_examples):
            y_guess = 0
            for count, weights in weights_vector:
                y_guess += count * calculate_guess(example, weights)  # get our guess with our weights
            y_guess = -1 if y_guess < 0 else 1
            y_actual = labels[example_index]  # get the actual labels
            if y_guess != y_actual:  # if they don't match up increase the error count
                error_count += 1
        return error_count / numb_examples  # calculate error

    @staticmethod
    def check_avg_perceptron_error(testing_examples, labels, averages):
        error_count = 0  # a count for how many examples we get wrong
        numb_examples = len(testing_examples)  # how many examples we have
        for example_index, example in enumerate(testing_examples):
            y_guess = calculate_guess(example, averages)  # get our guess with our weights
            y_actual = labels[example_index]  # get the actual labels
            if y_guess != y_actual:  # if they don't match up increase the error count
                error_count += 1
        return error_count / numb_examples  # calculate error
from ID3 import *
from math import exp, log

class Adaboost:
    def __init__(self, training_examples, test_examples, attributes, numb_iterations):
        self.numb_classifiers = numb_iterations
        self.label_dict = Adaboost.relabel_labels(training_examples, test_examples, attributes)
        #self.votes, self.classifiers, self.train_errors, self.test_errors = Adaboost.adaboost(training_examples, test_examples, attributes, numb_iterations)
        self.train_errors, self.test_errors = Adaboost.adaboost(training_examples, test_examples, attributes, numb_iterations)

    # def check_error(self, examples, attributes):
    #     numb_examples = len(examples)
    #     numb_incorrect = 0
    #     for example in examples:
    #         label_acutal = example[-1]
    #         label_final_guess = 0
    #         for index in range(self.numb_classifiers):
    #             classifier = self.classifiers[index]
    #             vote = self.votes[index]
    #             label_guess = classifier.travel_tree(example, attributes)
    #             label_final_guess += (vote * label_guess)
    #         # print(label_final_guess)
    #         sign = 1 if label_final_guess >= 0 else -1
    #         incorrect = (-(label_acutal * sign) + 1)/2
    #         numb_incorrect += incorrect
    #     return numb_incorrect / numb_examples
    
    @staticmethod
    def check_error_on_fly(classifier, vote, examples, attributes, example_previous_guesses):
        numb_incorrect = 0
        numb_examples = len(examples)
        new_guesses = []
        for index in range(numb_examples):
            example = examples[index]
            previous_guess = example_previous_guesses[index]
            label_guess = classifier.travel_tree(example, attributes)
            previous_guess += (vote * label_guess)
            new_guesses.append(previous_guess)
            sign = 1 if previous_guess >= 0 else -1
            label_actual = example[-1]
            incorrect = incorrect = (-(label_actual * sign) + 1)/2
            numb_incorrect += incorrect
        return numb_incorrect / numb_examples, new_guesses

    @staticmethod
    def adaboost(training_examples, test_examples, attributes, numb_iterations):
        previous_training_guesses = [0] * len(training_examples)
        previous_test_guesses = [0] * len(training_examples)
        test_errors = []
        train_errors = []
        numb_examples = len(training_examples)
        weights = Node.construct_weights(training_examples, 1/numb_examples)
        for i in range(numb_iterations):
            classifier = Node.decision_stump(training_examples, weights, attributes, Node.entropy)
            training_weighted_error, guesses = Adaboost.compute_weighted_error(classifier, training_examples, attributes, weights)
            alpha = 1 if training_weighted_error == 0 else (1/2) * log((1 - training_weighted_error)/training_weighted_error)
            weights = Adaboost.new_weights(weights, alpha, training_examples, guesses)
            train_error, previous_training_guesses = Adaboost.check_error_on_fly(classifier, alpha, training_examples, attributes, previous_training_guesses)
            test_error, previous_test_guesses = Adaboost.check_error_on_fly(classifier, alpha, test_examples, attributes, previous_test_guesses)
            train_errors.append(train_error)
            test_errors.append(test_error)
            Node.reclaim_attributes(attributes)
        return train_errors, test_errors

    @staticmethod
    def compute_weighted_error(classifier, training_examples, attributes, weights):
        guesses = []
        error = 0
        for example_index, example in enumerate(training_examples):
            guess = classifier.travel_tree(example, attributes)
            guesses.append(guess)
            example_label = example[-1]
            if example_label != guess:
                error += weights[example_index]
        error = error / sum(weights)
        return error, guesses

    @staticmethod
    def new_weights(weights, alpha, training_examples, guesses):
        new_weights = []
        for example_index, example in enumerate(training_examples):
            normalized_weight = weights[example_index]
            new_weight = normalized_weight * exp(-alpha * guesses[example_index] * example[-1])
            new_weights.append(new_weight)
        normalization_constant = sum(new_weights)
        return [weight/normalization_constant for weight in new_weights]

    @staticmethod
    def relabel_labels(training_examples, test_examples, attributes):
        labels = Node.find_label(attributes)
        label_values = labels.values
        new_label_dict = {-1: label_values[0], 1: label_values[1]}
        Adaboost.relabel_examples(training_examples, new_label_dict)
        Adaboost.relabel_examples(test_examples, new_label_dict)
        label_values[0] = -1
        label_values[1] = 1
        return new_label_dict

    @staticmethod
    def relabel_examples(examples, label_dict):
        for example in examples:
            if label_dict[-1] == example[-1]:  # comparing the new label of -1 with the label in the example
                example[-1] = -1
            else:
                example[-1] = 1

class Bagging:
    def __init__(self):
        pass

    
    pass
from ID3 import *
from math import exp, log
import random


def relabel_examples(examples, label_dict):
    for example in examples:
        if label_dict[-1] == example[-1]:  # comparing the new label of -1 with the label in the example
            example[-1] = -1
        else:
            example[-1] = 1

def relabel_labels(training_examples, test_examples, attributes):
    labels = Node.find_label(attributes)
    label_values = labels.values
    new_label_dict = {-1: label_values[0], 1: label_values[1]}
    relabel_examples(training_examples, new_label_dict)
    relabel_examples(test_examples, new_label_dict)
    label_values[0] = -1
    label_values[1] = 1
    return new_label_dict

class Adaboost:
    def __init__(self, training_examples, test_examples, attributes, numb_iterations):
        self.numb_classifiers = numb_iterations
        self.label_dict = relabel_labels(training_examples, test_examples, attributes)
        #self.votes, self.classifiers, self.train_errors, self.test_errors = Adaboost.adaboost(training_examples, test_examples, attributes, numb_iterations)
        self.train_errors, self.test_errors, self.training_errors_stump, self.testing_errors_stump = Adaboost.adaboost(training_examples, test_examples, attributes, numb_iterations)

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
        #classifiers = []
        #votes = []
        test_errors = []
        train_errors = []
        train_errors_stump = []
        test_errors_stump = []
        numb_examples = len(training_examples)
        weights = Node.construct_weights(training_examples, 1/numb_examples)
        for i in range(numb_iterations):
            print(i)
            classifier = Node.decision_stump(training_examples, weights, attributes, Node.entropy)
            training_weighted_error, guesses = Adaboost.compute_weighted_error(classifier, training_examples, attributes, weights)
            alpha = 1 if training_weighted_error == 0 else (1/2) * log((1 - training_weighted_error)/training_weighted_error)
            weights = Adaboost.new_weights(weights, alpha, training_examples, guesses)
            train_error, previous_training_guesses = Adaboost.check_error_on_fly(classifier, alpha, training_examples, attributes, previous_training_guesses)
            test_error, previous_test_guesses = Adaboost.check_error_on_fly(classifier, alpha, test_examples, attributes, previous_test_guesses)
            train_errors_stump.append(Node.check_error(classifier, training_examples, attributes))
            test_errors_stump.append(Node.check_error(classifier, test_examples, attributes))
            train_errors.append(train_error)
            test_errors.append(test_error)
            Node.reclaim_attributes(attributes)
        return train_errors, test_errors, train_errors_stump, test_errors_stump

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

class Bagging:
    def __init__(self, training_examples, test_examples, attributes, numb_iterations):
        self.label_dict = relabel_labels(training_examples, test_examples, attributes)
        self.training_errors, self.testing_errors, self.trees = Bagging.bagging(training_examples, test_examples, attributes, numb_iterations)
    
    @staticmethod
    def bagging(training_examples, test_examples, attributes, numb_iterations):
        classifiers = []
        training_errors = []
        test_errors = []
        numb_training_examples = len(training_examples)
        votes_training_examples = Bagging.track_votes_for_examples(training_examples, attributes)
        votes_test_examples = Bagging.track_votes_for_examples(test_examples, attributes)
        weights = [1] * len(training_examples)
        for i in range(numb_iterations):
            print(i)
            drawn_examples = random.choices(training_examples, k=numb_training_examples)
            max_depth = len(training_examples[0]) + 1
            tree = Node.id3(drawn_examples, weights, attributes, Node.entropy, 0, max_depth)
            Node.reclaim_attributes(attributes)
            train_error = Bagging.check_error_on_fly(tree, training_examples, votes_training_examples, attributes)
            test_error = Bagging.check_error_on_fly(tree, test_examples, votes_test_examples, attributes)
            classifiers.append(tree)
            training_errors.append(train_error)
            test_errors.append(test_error)
        return training_errors, test_errors, classifiers

    @staticmethod
    def track_votes_for_examples(examples, attributes):
        label_attr = Node.find_label(attributes)
        votes_examples = []
        for _ in examples:
            votes = {}
            for val in label_attr.values:
                votes[val] = 0
            votes_examples.append(votes)
        return votes_examples
    
    @staticmethod
    def check_error_on_fly(tree, examples, votes_examples, attributes):
        count_incorrect = 0
        for i in range(len(examples)):
            example = examples[i]
            tree_guess = tree.travel_tree(example, attributes)
            votes_example = votes_examples[i]
            votes_example[tree_guess] += 1
            overall_guess = max(votes_example, key=votes_example.get)
            if overall_guess != example[-1]:
                count_incorrect += 1
        return count_incorrect / len(examples)

            
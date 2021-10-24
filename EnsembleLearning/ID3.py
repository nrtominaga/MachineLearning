import statistics
from math import log2
import itertools


class Attribute:
    def __init__(self, name, values, is_label):
        self.attribute_name = name
        self.in_attributes = True
        self.values = values
        self.is_label = is_label

    def __str__(self):
        return 'name:' + Attribute.printable(self.attribute_name)

    @staticmethod
    def printable(thing):
        return thing + '\n'


class Node:

    def __init__(self, value, is_leaf):
        self.value = value
        self.children = {}
        self.is_leaf = is_leaf

    @staticmethod
    def check_error(tree, examples, attributes):
        correct_count = 0
        for example in examples:
            label_from_tree = tree.travel_tree(example, attributes)
            if label_from_tree == example[-1]:
                correct_count += 1
        return correct_count/len(examples)

    def travel_tree(self, example, attributes):
        if self.is_leaf:
            return self.value
        attr_split = self.value
        index, values = Node.find_attribute_values(attr_split, attributes)
        value = example[index]
        return self.children[value].travel_tree(example, attributes)

    @staticmethod
    def same_label(examples):  # check if all labels are the same
        first_label = examples[0][-1]
        for example in examples[1:]:
            if example[-1] != first_label:
                return False, ''
        return True, first_label

    @staticmethod
    def get_majority_label(examples, weights, attributes):
        label = attributes[-1]
        label_values = {}
        for value in label.values:
            label_values[value] = 0
        majority_label = label.values[0]
        majority_label_count = 0
        for example_index, example in enumerate(examples):
            label = example[-1]
            label_values[label] += weights[example_index]
            if label_values[label] > majority_label_count:
                majority_label_count = label_values[label]
                majority_label = label
        return majority_label

    @staticmethod
    def decision_stump(examples, weights, attributes, info_gain_method):
        return Node.id3(examples, weights, attributes, info_gain_method, 0, 1)

    @staticmethod
    def id3(examples, weights, attributes, info_gain_method, current_depth, max_depth):  # weights
        if max_depth == current_depth or Node.attributes_empty(attributes):
            value = Node.get_majority_label(examples, weights, attributes)
            return Node(value, True)
        is_same_label, value = Node.same_label(examples)
        if is_same_label:
            return Node(value, True)
        else:
            attribute_split = Node.info_gain(examples, attributes, weights, info_gain_method)
            new_root = Node(attribute_split, False)
            attribute_index, attribute_values = Node.find_attribute_values(attribute_split, attributes)
            for value in attribute_values:
                subset, subset_weights = Node.find_subset_examples(value, attribute_index, examples, weights)
                if len(subset) == 0:
                    new_value = Node.get_majority_label(examples, weights, attributes)
                    new_root.children[value] = Node(new_value, True)
                else:
                    attributes[attribute_index].in_attributes = False
                    new_root.children[value] = Node.id3(subset, subset_weights, attributes, info_gain_method,
                                                        current_depth + 1, max_depth)
                    attributes[attribute_index].in_attributes = True
            return new_root

    @staticmethod
    def find_label(attributes):
        for attribute in attributes:
            if attribute.is_label:
                return attribute

    @staticmethod
    def find_attribute_values(attribute_name, attributes):
        for i, attribute in enumerate(attributes):
            if attribute.attribute_name == attribute_name and attribute.in_attributes:
                return i, attribute.values

    @staticmethod
    def find_subset_examples(attribute_value, index, examples, weights):
        new_subset = []
        subset_weights = []
        for example_index, example in enumerate(examples):
            if example[index] == attribute_value:
                new_subset.append(example)
                subset_weights.append(weights[example_index])
        return new_subset, subset_weights

    @staticmethod
    def reclaim_attributes(attributes):
        for attribute in attributes:
            attribute.in_attributes = True

    @staticmethod
    def attributes_empty(attributes):
        for attribute in attributes:
            if attribute.in_attributes and not attribute.is_label:
                return False
        return True

    @staticmethod
    def open_data(filename):
        examples = []  # represents our set of examples
        with open(filename) as data:
            for example in data:
                example = example.replace('\n', '').split(
                    ',')  # replace each end line with an empty string and then split
                examples.append(example)  # add to set of examples
        return examples

    @staticmethod
    def create_attributes(names, vals_to_create, attr_training_examples, attr_test_examples):
        attributes_to_create = []
        for create_index, name in enumerate(names):
            if vals_to_create[create_index] == 'numeric':
                attribute_vals = []
                for example in attr_training_examples:
                    attribute_vals.append(int(example[create_index]))
                median = statistics.median(attribute_vals)
                for example in itertools.chain(attr_training_examples, attr_test_examples):
                    if int(example[create_index]) <= median:
                        example[create_index] = 'le'
                    else:
                        example[create_index] = 'g'
                vals_to_create[create_index] = ['le', 'g']
            new_attribute = Attribute(name, vals_to_create[create_index], False)
            attributes_to_create.append(new_attribute)
        attributes_to_create[-1].is_label = True
        return attributes_to_create

    @staticmethod
    def construct_weights(training_examples, initial_value):
        return [initial_value] * len(training_examples)

    @staticmethod
    def replace_unknowns(unk_training_examples, unk_test_examples, unk_attributes):
        for unknown_index, attribute in enumerate(unk_attributes):
            if 'unknown' in attribute.values:
                values_count = dict.fromkeys(attribute.values, 0)
                del values_count['unknown']
                max_count = 0
                max_label = unk_training_examples[0][unknown_index]
                for example in unk_training_examples:
                    if example[unknown_index] != 'unknown':
                        values_count[example[unknown_index]] += 1
                        if values_count[example[unknown_index]] > max_count:
                            max_label = example[unknown_index]
                for example in unk_training_examples:
                    if example[unknown_index] == 'unknown':
                        example[unknown_index] = max_label
                for example in unk_test_examples:
                    if example[unknown_index] == 'unknown':
                        example[unknown_index] = max_label

    @staticmethod
    def info_gain(examples, ig_attributes, weights, info_gain_method):
        label = Node.find_label(ig_attributes)  # label attribute object
        entropy_s = info_gain_method(examples, label, weights)
        greatest_gain = 0
        greatest_gain_name = ig_attributes[0].attribute_name
        for ig_i, attribute in enumerate(ig_attributes):
            if not attribute.is_label and attribute.in_attributes:
                attribute_gain = 0
                for value in attribute.values:
                    subset_examples, subset_weights = Node.find_subset_examples(value, ig_i, examples, weights)
                    value_count = sum(subset_weights)
                    proportion_value = value_count / sum(weights)
                    if value_count != 0:
                        attribute_gain += (proportion_value * info_gain_method(subset_examples, label,
                                                                               subset_weights))
                gain = entropy_s - attribute_gain
                if gain >= greatest_gain:
                    greatest_gain = gain
                    greatest_gain_name = attribute.attribute_name
        return greatest_gain_name

    @staticmethod
    def entropy(examples, label, weights):
        numb_examples = sum(weights)
        final_entropy = 0
        for value in label.values:
            label_count = 0
            for example_index, example in enumerate(examples):
                if example[-1] == value:
                    label_count += weights[example_index]
            p = label_count / numb_examples
            if p != 0:
                final_entropy += (-p * log2(p))
        return final_entropy

    @staticmethod
    def majority_error(examples, label, weights):
        label_counts = dict.fromkeys(label.values, 0)
        max_count = 0
        max_label = label.values[0]
        for example_index, example in enumerate(examples):
            label_counts[example[-1]] += weights[example_index]
            if max_count < label_counts[example[-1]]:
                max_count = label_counts[example[-1]]
                max_label = example[-1]
        wrong_count = 0
        for example_index, example in enumerate(examples):
            if example[-1] != max_label:
                wrong_count += weights[example_index]
        return wrong_count / sum(weights)

    @staticmethod
    def gini_index(examples, label, weights):
        numb_examples = sum(weights)
        label_counts = dict.fromkeys(label.values, 0)
        for example_index, example in enumerate(examples):
            label_counts[example[-1]] += weights[example_index]
        gi = 1
        for label_count in label_counts:
            gi -= ((label_counts[label_count] / numb_examples) ** 2)
        return gi
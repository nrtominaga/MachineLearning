import statistics
from math import log2


class Attribute:
    def __init__(self, name, values, is_label, is_numeric, median):
        self.attribute_name = name
        self.in_attributes = True
        self.values = values
        self.is_label = is_label
        self.is_numeric = is_numeric
        self.median = median

    def __str__(self):
        return 'name:' + Attribute.printable(self.attribute_name)

    @staticmethod
    def printable(thing):
        return thing + '\n'


class Node:

    count = 0

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
        return correct_count/len(examples) * 100

    def travel_tree(self, example, attributes):
        if self.is_leaf:
            return self.value
        attr_split = self.value
        index, values = Node.find_attribute_values(attr_split, attributes)
        value = example[index]
        if attributes[index].is_numeric:
            if value <= attributes[index].median:
                return self.children['le'].travel_tree(example, attributes)
            else:
                return self.children['g'].travel_tree(example, attributes)
        else:
            return self.children[value].travel_tree(example, attributes)

    @staticmethod
    def same_label(examples):  # check if all labels are the same
        first_label = examples[0][-1]
        for example in examples[1:]:
            if example[-1] != first_label:
                return False, ''
        return True, first_label

    @staticmethod
    def get_majority_label(examples, attributes):
        label = attributes[-1]
        label_values = {}
        for value in label.values:
            label_values[value] = 0
        majority_label = label.values[0]
        majority_label_count = 0
        for example in examples:
            label = example[-1]
            label_values[label] += 1
            if label_values[label] > majority_label_count:
                majority_label_count = label_values[label]
                majority_label = label
        return majority_label

    @staticmethod
    def decision_stump(examples, attributes, info_gain):
        return Node.id3(examples, attributes, info_gain, 0, 1)

    @staticmethod
    def id3(examples, attributes, info_gain, current_depth, max_depth):
        if max_depth == current_depth or Node.attributes_empty(attributes):
            value = Node.get_majority_label(examples, attributes)
            return Node(value, True)
        is_same_label, value = Node.same_label(examples)
        if is_same_label:
            return Node(value, True)
        else:
            attribute_split = info_gain(examples, attributes)
            new_root = Node(attribute_split, False)
            attribute_index, attribute_values = Node.find_attribute_values(attribute_split, attributes)
            for value in attribute_values:
                subset = Node.find_subset_examples(value, attribute_index, examples, attributes)
                if len(subset) == 0:
                    new_value = Node.get_majority_label(examples, attributes)
                    new_root.children[value] = Node(new_value, True)
                else:
                    attributes[attribute_index].in_attributes = False
                    new_root.children[value] = Node.id3(subset, attributes, info_gain, current_depth+1, max_depth)
                    attributes[attribute_index].in_attributes = True
            return new_root

    @staticmethod
    def find_label(attributes):
        for attribute in attributes:
            if attribute.is_label:
                return attribute

    @staticmethod
    def find_attribute_values(attribute_name, attributes):
        Node.count += 1
        # print(Node.count)
        for i, attribute in enumerate(attributes):
            if attribute.attribute_name == attribute_name and attribute.in_attributes:
                return i, attribute.values

    @staticmethod
    def find_subset_examples(attribute_value, index, examples, attributes):
        new_subset = []
        for example in examples:
            if attributes[index].is_numeric:
                if attribute_value == 'le' and example[index] <= attributes[index].median:
                    new_subset.append(example)
                elif attribute_value == 'g' and example[index] > attributes[index].median:
                    new_subset.append(example)
            elif example[index] == attribute_value:
                new_subset.append(example)
        return new_subset

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
                    example[create_index] = int(example[create_index])
                    attribute_vals.append(example[create_index])
                median = statistics.median(attribute_vals)
                for example in attr_test_examples:
                    example[create_index] = int(example[create_index])
                new_attribute = Attribute(name, ['le', 'g'], False, True, median)
                attributes_to_create.append(new_attribute)
            else:
                new_attribute = Attribute(name, vals_to_create[create_index], False, False, -1)
                attributes_to_create.append(new_attribute)
        attributes_to_create[-1].is_label = True
        return attributes_to_create

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
    def info_gain(examples, ig_attributes):
        label = Node.find_label(ig_attributes)  # label attribute object
        numb_examples = len(examples)  # number of examples
        entropy_s = 0
        for value in label.values:
            label_count = 0
            for example in examples:
                if example[-1] == value:
                    label_count += 1
            p = label_count / numb_examples
            if p != 0:
                entropy_s += (-p * log2(p))
        greatest_gain = 0
        greatest_gain_name = ig_attributes[0].attribute_name
        for ig_i, attribute in enumerate(ig_attributes):
            if not attribute.is_label and attribute.in_attributes:
                expected_entropy = 0
                if attribute.is_numeric:
                    values_numeric = {'g': 0, 'le': 0}
                    label_values_numeric = dict.fromkeys(label.values, 0)
                    for example in examples:
                        if example[ig_i] <= attribute.median:
                            values_numeric['le'] += 1
                            label_values_numeric[example[-1]] += 1
                    entropy_value_numeric = 0
                    if values_numeric['le'] != 0:
                        for label_value in label_values_numeric:
                            p = label_values_numeric[label_value] / values_numeric['le']
                            if p != 0:
                                entropy_value_numeric += (-p * log2(p))
                        expected_entropy += ((values_numeric['le'] / len(examples)) * entropy_value_numeric)
                    label_values_numeric = dict.fromkeys(label.values, 0)
                    for example in examples:
                        if example[ig_i] > attribute.median:
                            values_numeric['g'] += 1
                            label_values_numeric[example[-1]] += 1
                    entropy_value_numeric = 0
                    if values_numeric['g'] != 0:
                        for label_value in label_values_numeric:
                            p = label_values_numeric[label_value] / values_numeric['g']
                            if p != 0:
                                entropy_value_numeric += (-p * log2(p))
                        expected_entropy += ((values_numeric['g'] / len(examples)) * entropy_value_numeric)
                else:
                    for value in attribute.values:
                        label_values = dict.fromkeys(label.values, 0)
                        value_count = 0
                        for example in examples:
                            if example[ig_i] == value:
                                value_count += 1
                                label_values[example[-1]] += 1
                        proportion_value = value_count / numb_examples
                        entropy_value = 0
                        if value_count != 0:
                            for label_value in label_values:
                                p = label_values[label_value] / value_count
                                if p != 0:
                                    entropy_value += (-p * log2(p))
                            expected_entropy += (proportion_value * entropy_value)
                gain = entropy_s - expected_entropy
                if gain >= greatest_gain:
                    greatest_gain = gain
                    greatest_gain_name = attribute.attribute_name
        return greatest_gain_name

    @staticmethod
    def majority_error(examples, me_attributes):
        label = Node.find_label(me_attributes)  # label attribute object
        label_counts = dict.fromkeys(label.values, 0)
        max_count = 0
        max_label = label.values[0]
        for example in examples:
            label_counts[example[-1]] += 1
            if max_count < label_counts[example[-1]]:
                max_count = label_counts[example[-1]]
                max_label = example[-1]
        wrong_count = 0
        for example in examples:
            if example[-1] != max_label:
                wrong_count += 1
        numb_examples = len(examples)  # number of examples
        maj_error_s = wrong_count / numb_examples
        greatest_gain = 0
        greatest_gain_name = me_attributes[0].attribute_name
        for me_i, attribute in enumerate(me_attributes):
            if not attribute.is_label and attribute.in_attributes:
                maj_error_val = 0
                if attribute.is_numeric:
                    values_numeric = {'g': 0, 'le': 0}
                    label_values_numeric = dict.fromkeys(label.values, 0)
                    maj_label = label.values[0]
                    maj_count = 0
                    for example in examples:
                        if example[me_i] <= attribute.median:
                            values_numeric['le'] += 1
                            label_values_numeric[example[-1]] += 1
                            if label_values_numeric[example[-1]] > maj_count:
                                maj_count = label_values_numeric[example[-1]]
                                maj_label = example[-1]
                    error_count = 0
                    for numeric_label in label_values_numeric:
                        if numeric_label != maj_label:
                            error_count += label_values_numeric[numeric_label]
                    if label_values_numeric[maj_label] != 0:
                        maj_error_val += ((label_values_numeric[maj_label] / numb_examples) *
                                          (error_count / label_values_numeric[maj_label]))
                    maj_label = label.values[0]
                    maj_count = 0
                    label_values_numeric = dict.fromkeys(label.values, 0)
                    for example in examples:
                        if example[me_i] > attribute.median:
                            values_numeric['g'] += 1
                            label_values_numeric[example[-1]] += 1
                            if label_values_numeric[example[-1]] > maj_count:
                                maj_count = label_values_numeric[example[-1]]
                                maj_label = example[-1]
                    error_count = 0
                    for numeric_label in label_values_numeric:
                        if numeric_label != maj_label:
                            error_count += label_values_numeric[numeric_label]
                    if label_values_numeric[maj_label] != 0:
                        maj_error_val += ((label_values_numeric[maj_label] / numb_examples) *
                                          (error_count / label_values_numeric[maj_label]))
                else:
                    attribute_vals = attribute.values
                    for value in attribute_vals:
                        label_values = dict.fromkeys(label.values, 0)
                        maj_label = label.values[0]
                        maj_label_count = 0
                        for example in examples:
                            if example[me_i] == value:
                                label_values[example[-1]] += 1
                                if label_values[example[-1]] >= maj_label_count:
                                    maj_label_count = label_values[example[-1]]
                                    maj_label = example[-1]
                        error_count = 0
                        for label_value in label_values:
                            if label_value != maj_label:
                                error_count += label_values[label_value]
                        if label_values[maj_label] != 0:
                            maj_error_val += ((label_values[maj_label] / numb_examples) *
                                              (error_count / label_values[maj_label]))
                gain = maj_error_s - maj_error_val
                if gain >= greatest_gain:
                    greatest_gain = gain
                    greatest_gain_name = attribute.attribute_name
        return greatest_gain_name

    @staticmethod
    def gini_index(examples, gini_index_attributes):
        label = Node.find_label(gini_index_attributes)  # label attribute object
        label_counts = dict.fromkeys(label.values, 0)
        numb_examples = len(examples)
        for example in examples:
            label_counts[example[-1]] += 1
        gi_s = 1
        for label_count in label_counts:
            gi_s -= ((label_counts[label_count] / numb_examples) ** 2)
        greatest_gain = 0
        greatest_gain_name = gini_index_attributes[0].attribute_name
        for gini_index_i, attribute in enumerate(gini_index_attributes):
            if not attribute.is_label and attribute.in_attributes:
                expected_gain = 0
                if attribute.is_numeric:
                    values_numeric = {'g': 0, 'le': 0}
                    label_values_numeric = dict.fromkeys(label.values, 0)
                    for example in examples:
                        if example[gini_index_i] <= attribute.median:
                            values_numeric['le'] += 1
                            label_values_numeric[example[-1]] += 1
                    gi_val = 1
                    for label_value_numeric in label_values_numeric:
                        if values_numeric['le'] != 0:
                            gi_val -= ((label_values_numeric[label_value_numeric] / values_numeric['le']) ** 2)
                    expected_gain += ((values_numeric['le'] / numb_examples) * gi_val)
                    label_values_numeric = dict.fromkeys(label.values, 0)
                    for example in examples:
                        if example[gini_index_i] > attribute.median:
                            values_numeric['g'] += 1
                            label_values_numeric[example[-1]] += 1
                    gi_val = 1
                    for label_value_numeric in label_values_numeric:
                        if values_numeric['g'] != 0:
                            gi_val -= ((label_values_numeric[label_value_numeric] / values_numeric['g']) ** 2)
                    expected_gain += ((values_numeric['g'] / numb_examples) * gi_val)
                else:
                    attribute_vals = attribute.values
                    for value in attribute_vals:
                        label_values = dict.fromkeys(label.values, 0)
                        value_count = 0
                        for example in examples:
                            if example[gini_index_i] == value:
                                label_values[example[-1]] += 1
                                value_count += 1
                        gi_val = 1
                        for label_value in label_values:
                            if value_count != 0:
                                gi_val -= ((label_values[label_value] / value_count) ** 2)
                        expected_gain += ((value_count / numb_examples) * gi_val)
                gain = gi_s - expected_gain
                if gain >= greatest_gain:
                    greatest_gain = gain
                    greatest_gain_name = attribute.attribute_name
        return greatest_gain_name

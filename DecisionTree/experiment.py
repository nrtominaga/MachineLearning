from ID3 import *

car_attributes_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
car_attributes_vals = [['vhigh', 'high', 'med', 'low'], ['vhigh', 'high', 'med', 'low'], ['2', '3', '4', '5more'],
                       ['2', '4', 'more'], ['small', 'med', 'big'], ['low', 'med', 'high'],
                       ['unacc', 'acc', 'good', 'vgood']]

practice_attribute_names = ['O', 'T', 'H', 'W', 'Play']
practice_attribute_vals = [['S', 'O', 'R'], ['H', 'M', 'C'], ['H', 'N', 'L'], ['S', 'W'], ['+', '-']]

bank_attribute_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                        'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
bank_attribute_vals = ['numeric',
                       ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                        "blue-collar", "self-employed", "retired", "technician", "services"],
                       ["married", "divorced", "single"], ["unknown", "secondary", "primary", "tertiary"],
                       ["yes", "no"], 'numeric', ["yes", "no"], ["yes", "no"], ["unknown", "telephone", "cellular"],
                       'numeric', ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                       'numeric', 'numeric', 'numeric', 'numeric', ["unknown", "other", "failure", "success"],
                       ["yes", "no"]]


def to_string(_training_examples, _test_examples, _attributes, to_depth):
    string_train = ''
    string_test = ''
    for depth in range(1, to_depth + 1):
        string_train += '%d & ' % depth
        string_test += '%d & ' % depth
        tree = Node.id3(_training_examples, _attributes, Node.info_gain, 0, depth)
        Node.reclaim_attributes(_attributes)
        string_train += '%.2f & ' % Node.check_error(tree, _training_examples, _attributes)
        string_test += '%.2f & ' % Node.check_error(tree, _test_examples, _attributes)
        tree = Node.id3(_training_examples, _attributes, Node.majority_error, 0, depth)
        Node.reclaim_attributes(_attributes)
        string_train += '%.2f & ' % Node.check_error(tree, _training_examples, _attributes)
        string_test += '%.2f & ' % Node.check_error(tree, _test_examples, _attributes)
        tree = Node.id3(_training_examples, _attributes, Node.gini_index, 0, depth)
        Node.reclaim_attributes(_attributes)
        string_train += '%.2f ' % Node.check_error(tree, _training_examples, _attributes)
        string_test += '%.2f ' % Node.check_error(tree, _test_examples, _attributes)
        string_train += r'\\ \hline'
        string_test += r'\\ \hline'
        string_test += '\n\t\t'
        string_train += '\n\t\t'
    return string_train, string_test


if __name__ == "__main__":
    car_training_examples = Node.open_data('./Data/car/train.csv')  # load in training data
    car_test_examples = Node.open_data('./Data/car/test.csv')  # load in car test data
    car_attributes = Node.create_attributes(car_attributes_names, car_attributes_vals, car_training_examples,
                                       car_test_examples)
    car_string_train, car_string_test = to_string(car_training_examples, car_test_examples, car_attributes, 6)
    print('car training data:')
    print(car_string_train)
    print('car test data:')
    print(car_string_test)
    bank_training_examples = Node.open_data('./Data/bank/train.csv')
    bank_test_examples = Node.open_data('./Data/bank/test.csv')
    bank_attributes = Node.create_attributes(bank_attribute_names, bank_attribute_vals, bank_training_examples,
                                        bank_test_examples)
    bank_string_train, bank_string_test = to_string(bank_training_examples, bank_test_examples, bank_attributes, 16)
    print('bank training data:')
    print(bank_string_train)
    print('bank test data:')
    print(bank_string_test)
    Node.replace_unknowns(bank_training_examples, bank_test_examples, bank_attributes)
    bank_string_train, bank_string_test = to_string(bank_training_examples, bank_test_examples, bank_attributes, 16)
    print('bank training data without unknowns:')
    print(bank_string_train)
    print('bank test data without unknowns:')
    print(bank_string_test)


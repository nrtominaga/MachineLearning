from Ensemble_Learning import Adaboost as ada
from ID3 import Node as nd
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    training_examples = nd.open_data('./Data/bank/train.csv')
    test_examples = nd.open_data('./Data/bank/test.csv')
    attributes = nd.create_attributes(bank_attribute_names, bank_attribute_vals, training_examples, test_examples)
    adaboost = ada(training_examples, test_examples, attributes, 500)
    # training_errors = adaboost.check_error(training_examples, attributes)
    # print(training_errors) # 0.1028
    one_to_five_hundred = list(range(1,501))
    plt.plot(one_to_five_hundred, adaboost.train_errors, label="Training Error")
    plt.plot(one_to_five_hundred, adaboost.test_errors, label="Test Errors")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.title("Adaboost Training and Test Errors")
    plt.legend()

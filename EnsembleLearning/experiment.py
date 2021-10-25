from Ensemble_Learning import Adaboost as ada, Bagging as bg, Random_Forest as rf
from ID3 import Node as nd
import matplotlib.pyplot as plt
import random


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
    figures = 0
    # adaboost
    training_examples = nd.open_data('./Data/bank/train.csv')
    test_examples = nd.open_data('./Data/bank/test.csv')
    attributes = nd.create_attributes(bank_attribute_names, bank_attribute_vals, training_examples, test_examples)
    one_to_five_hundred = list(range(1,501))
    # adaboost = ada(training_examples, test_examples, attributes, 500)
    # # training_errors = adaboost.check_error(training_examples, attributes)
    # # print(training_errors) # 0.1028
    # plot1 = plt.figure(figures+=1)
    # plt.plot(one_to_five_hundred, adaboost.train_errors, label="Training Error")
    # plt.plot(one_to_five_hundred, adaboost.test_errors, label="Test Errors")
    # plt.xlabel("Iteration Number")
    # plt.ylabel("Error")
    # plt.title("Adaboost Training and Test Errors")
    # plt.legend()

    # plot2 = plt.figure(figures+=1)
    # plt.plot(one_to_five_hundred, adaboost.training_errors_stump, label="Training Error")
    # plt.plot(one_to_five_hundred, adaboost.testing_errors_stump, label="Test Errors")
    # plt.xlabel("Iteration Number")
    # plt.ylabel("Error")
    # plt.title("Decision Stump Training and Test Errors")
    # plt.legend()

    # # bagging
    # bagging = bg(training_examples, test_examples, attributes, 500)
    # plot3 = plt.figure(figures+=1)
    # plt.plot(one_to_five_hundred, bagging.training_errors, label="Training Error")
    # plt.plot(one_to_five_hundred, bagging.testing_errors, label="Test Errors")
    # plt.xlabel("Iteration Number")
    # plt.ylabel("Error")
    # plt.title("Bagging Training and Test Errors")
    # plt.legend()

    #bias and variance decomposition 
    # iterations = 100
    # numb_samples = 1000
    # numb_trees = 500
    # first_trees = []
    # for i in range(iterations):
    #     print(i)
    #     sub_sample = random.sample(training_examples, numb_samples)
    #     bagging = bg(sub_sample, test_examples, attributes, numb_trees)
    #     first_trees.append(bagging.trees[0])
    #     for example in test_examples:
    #         pass
    # single_tree_bias = 0
    # single_tree_var = 0
    # for example in test_examples:
    #     avg = 0
    #     predictions = []
    #     for tree in first_trees:
    #         p = tree.travel_tree(example, attributes)
    #         predictions.append(p)
    #         avg += p
    #     avg /= len(first_trees)
    #     var = 0
    #     for p in predictions:
    #         var += ((p - avg) ** 2)
    #     var /= (len(first_trees) - 1)
    #     single_tree_var += var
    #     bias = (avg - example[-1]) ** 2
    #     single_tree_bias += bias
    # print(single_tree_bias/len(test_examples))


    # bagging
    subset_sizes = [2,4,6]
    rf_training_errors = []
    rf_test_errors = []
    for s in subset_sizes:
        random_forest = rf(training_examples, test_examples, attributes, 500, s)
        rf_training_errors.append(random_forest.training_errors)
        rf_test_errors.append(random_forest.test_errors)

    figures+=1
    plot = plt.figure(figures)
    plt.plot(one_to_five_hundred, rf_training_errors[0], label="Subset Size = 2")
    plt.plot(one_to_five_hundred, rf_training_errors[1], label="Subset Size = 4")
    plt.plot(one_to_five_hundred, rf_training_errors[2], label="Subset Size = 6")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.title(f"Random Forest Training Errors")
    plt.legend()


    figures+=1
    plot = plt.figure(figures)
    plt.plot(one_to_five_hundred, rf_test_errors[0], label="Subset Size = 2")
    plt.plot(one_to_five_hundred, rf_test_errors[1], label="Subset Size = 4")
    plt.plot(one_to_five_hundred, rf_test_errors[2], label="Subset Size = 6")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.title(f"Random Forest Test Errors")
    plt.legend()

    plt.show()
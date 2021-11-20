import enum
from SVM import SVM, Dual_SVM
from fractions import Fraction 
import math
import numpy as np

training_file = './Data/bank-note/train.csv'
testing_file = './Data/bank-note/test.csv'

Cs = ['100/873', '500/873', '700/873']

a_vals = [1, 5, 10, 50, 100, 500, 1000]
gammas = [.005, .001, .0005, .0001]

if __name__ == "__main__":
    # print('5-fold CV') # perform 5 fold CV
    # print('On a vals: ', a_vals)
    # print('On gamma vals:', gammas)
    # best_error = math.inf
    # best_a = None
    # best_gamma = None
    # for a in a_vals:
    #     for gamma in gammas:
    #         ksvm = SVM(training_file, testing_file, 100, gamma, float(Fraction(Cs[1])), a, 5)
    #         if ksvm.avg_error < best_error:
    #             best_error = ksvm.avg_error
    #             best_a = a
    #             best_gamma = gamma
    # print('Best a val: ', best_a)
    # print('Best gamma val: ', best_gamma)
    # print('Stochastic Sub-Gradient Descent\n') # 2
    # print('Schedule Learning Rate: gamme_0/(1+(gamma_0/a)*t)') # part a
    # for C in Cs:
    #     svm = SVM(training_file, testing_file, 100, best_gamma, float(Fraction(C)), a=best_a)
    #     print('C = ', C)
    #     print('weights = ', svm.weights)
    #     print('training error = ', SVM.error(svm.weights, svm.training_examples, svm.training_labels))
    #     print('testing error = ', SVM.error(svm.weights, svm.testing_examples, svm.testing_labels))
    # print('\nSchedule Learning Rate: gamme_0/(1+t)') # part b
    # for C in Cs:
    #     svm = SVM(training_file, testing_file, 100, best_gamma, float(Fraction(C)))
    #     print('C = ', C)
    #     print('weights = ', svm.weights)
    #     print('training error = ', SVM.error(svm.weights, svm.training_examples, svm.training_labels))
    #     print('testing error = ', SVM.error(svm.weights, svm.testing_examples, svm.testing_labels))

    # print('\nDual SVM w/ Linear Kernel\n')
    # for C in Cs:
    #     print('C: ', C)
    #     svm = Dual_SVM(training_file, testing_file, float(Fraction(C)))
    #     print('Weights: ', svm.weights)
    #     print('Training error: ', svm.error('training'))
    #     print('Testing error: ', svm.error('testing'))

    gammas = [0.1, 0.5, 1, 5, 100]
    print('\nDual SVM w/ Gaussian Kernel\n')
    smallest_testing_error = math.inf
    smallest_error_gamma = gammas[0]
    smallest_error_C = Cs[0]
    sv_dict = {}
    for C in Cs:
        for g in gammas:
            print('C: ', C, 'gamma: ', g)
            svm = Dual_SVM(training_file, testing_file, float(Fraction(C)), g)
            svm.alphas[np.isclose(svm.alphas, 0)] = 0
            print('Weights: ', svm.weights)
            print('Training error: ', svm.error('training'))
            testing_error = svm.error('testing')
            print('Testing error: ', testing_error)
            if testing_error < smallest_testing_error:
                smallest_testing_error = testing_error
                smallest_error_gamma = g
                smallest_error_C = C

            is_support_vector = svm.alphas > 0
            print('Number of support vectors:', np.sum(is_support_vector))

            if C == '500/873':
                is_support_vector = svm.alphas > 0
                sv_dict[g] = is_support_vector

    print('Best pair C: ', smallest_error_C, ' gamma: ', smallest_error_gamma)

    print('\nNumber of overlapping supoort vectors:')
    for g_i, g1 in enumerate(gammas):
        for _, g2 in enumerate(gammas, g_i + 1):
            numb_ov = np.sum(sv_dict[g1] & sv_dict[g2])
            print('Number of overlapped vectors between', g1, 'and', g2 , ':', numb_ov)
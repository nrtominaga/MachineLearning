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
    with open('output.txt', 'w+') as file:
        file.write('5-fold CV' + '\n') # perform 5 fold CV
        file.write('On a vals: ' +  str(a_vals)+ '\n')
        file.write('On gamma vals: ' + str(gammas)+ '\n')
        best_error = math.inf
        best_a = None
        best_gamma = None
        for a in a_vals:
            for gamma in gammas:
                ksvm = SVM(training_file, testing_file, 100, gamma, float(Fraction(Cs[1])), a, 5)
                if ksvm.avg_error < best_error:
                    best_error = ksvm.avg_error
                    best_a = a
                    best_gamma = gamma
        file.write('Best a val: ' + str(best_a)+ '\n')
        file.write('Best gamma val: ' + str(best_gamma) + '\n')
        file.write('\n' + 'Stochastic Sub-Gradient Descent\n') # 2
        file.write('Schedule Learning Rate: gamme_0/(1+(gamma_0/a)*t)'+ '\n') # part a
        for C in Cs:
            svm = SVM(training_file, testing_file, 100, best_gamma, float(Fraction(C)), a=best_a)
            file.write('C = ' + str(C)+ '\n')
            file.write('weights = ' + str(svm.weights)+ '\n')
            file.write('training error = ' + str(SVM.error(svm.weights, svm.training_examples, svm.training_labels))+ '\n')
            file.write('testing error = ' + str(SVM.error(svm.weights, svm.testing_examples, svm.testing_labels))+ '\n')
        file.write('\nSchedule Learning Rate: gamme_0/(1+t)'+ '\n') # part b
        for C in Cs:
            svm = SVM(training_file, testing_file, 100, best_gamma, float(Fraction(C)))
            file.write('C = ' + str(C)+ '\n')
            file.write('weights = ' + str(svm.weights)+ '\n')
            file.write('training error = ' +  str(SVM.error(svm.weights, svm.training_examples, svm.training_labels))+ '\n')
            file.write('testing error = ' + str(SVM.error(svm.weights, svm.testing_examples, svm.testing_labels))+ '\n')

        file.write('\nDual SVM w/ Linear Kernel\n')
        for C in Cs:
            file.write('C: ' + str(C) + '\n')
            svm = Dual_SVM(training_file, testing_file, float(Fraction(C)))
            file.write('Weights: ' + str(svm.weights) + '\n')
            file.write('Training error: ' + str(svm.error('training'))+ '\n')
            file.write('Testing error: ' + str(svm.error('testing'))+ '\n')

        gammas = [0.1, 0.5, 1, 5, 100]
        file.write('\nDual SVM w/ Gaussian Kernel\n')
        smallest_testing_error = math.inf
        smallest_error_gamma = gammas[0]
        smallest_error_C = Cs[0]
        sv_dict = {}
        for C in Cs:
            for g in gammas:
                file.write('C: ' + str(C) + ' gamma: ' + str(g)+ '\n')
                svm = Dual_SVM(training_file, testing_file, float(Fraction(C)), g)
                svm.alphas[np.isclose(svm.alphas, 0)] = 0
                file.write('Training error: ' + str(svm.error('training'))+ '\n')
                testing_error = svm.error('testing')
                file.write('Testing error: ' + str(testing_error)+ '\n')
                if testing_error < smallest_testing_error:
                    smallest_testing_error = testing_error
                    smallest_error_gamma = g
                    smallest_error_C = C

                is_support_vector = svm.alphas > 0
                file.write('Number of support vectors: ' + str(np.sum(is_support_vector))+ '\n')

                if C == '500/873':
                    is_support_vector = svm.alphas > 0
                    sv_dict[g] = is_support_vector

        file.write('Best pair C: ' + str(smallest_error_C) + ' gamma: ' + str(smallest_error_gamma)+ '\n')
        file.write('\nError is much better than in the linear kernel probably because the guassian kernel is able to make a classifier in a higher dimensional space.\n')

        file.write('\nNumber of overlapping supoort vectors:\n')
        for g_i, g1 in enumerate(gammas):
            for g_j in range(g_i + 1, len(gammas)):
                g2 = gammas[g_j]
                numb_ov = np.sum(sv_dict[g1] & sv_dict[g2])
                file.write('Number of overlapped vectors between ' + str(g1) + ' and ' + str(g2) + ': ' + str(numb_ov)+ '\n')

        file.write('\n\nIt seems like in general that the higher the learning rate the smaller number of overlapping support vectors there are. Not entierly sure why.')
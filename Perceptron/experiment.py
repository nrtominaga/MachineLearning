from Perceptron import Perceptron

test_file = './Data/bank-note/test.csv'
train_file = './Data/bank-note/train.csv'


def write_weights_and_error(filename, weight, error):
    with open(filename, 'w+') as file:
        file.write(str(weight))
        file.write('\n')
        file.write(str(error))


# std perceptron
perceptron = Perceptron(train_file, test_file, 0.01, 10, Perceptron.perceptron)
write_weights_and_error('weights_and_error/std_perceptron.txt', perceptron.weights, perceptron.perform_error_function())
# voted perceptron
voted_perceptron = Perceptron(train_file, test_file, .01, 10, Perceptron.voted_perceptron)
write_weights_and_error('weights_and_error/voted_perceptron.txt', voted_perceptron.weights,
                        voted_perceptron.perform_error_function())
# averaged perceptron
avg_perceptron = Perceptron(train_file, test_file, .01, 10, Perceptron.averaged_perceptron)
write_weights_and_error('weights_and_error/avg_perceptron.txt', avg_perceptron.weights,
                        avg_perceptron.perform_error_function())
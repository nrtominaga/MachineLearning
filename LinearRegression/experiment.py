from Linear_Regression import Linear_Regression as lr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_file = "./Data/concrete/train.csv"
    test_file = "./Data/concrete/test.csv"

    bgd = lr(training_file, test_file, 1/256, lr.batch_gradient_descent)
    print("Final Weight Vector(bgd):")
    print(bgd.w)
    print("Final Test Cost(bgd):")
    print(lr.calculate_cost(bgd.test_examples, bgd.test_outputs, bgd.w))
    iterations = list(range(len(bgd.costs)))
    plot1 = plt.figure(1)
    plt.plot(iterations, bgd.costs, label="Costs")
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost")
    plt.title("Batch Gradient Descent Cost at Each Iteration")
    plt.legend()

    sgd = lr(training_file, test_file, 1/512, lr.stochastic_gradient_descent)
    print("Final Weight Vector(sgd):")
    print(sgd.w)
    print("Final Test Cost(sgd):")
    print(lr.calculate_cost(sgd.test_examples, sgd.test_outputs, sgd.w))
    iterations = list(range(len(sgd.costs)))
    plot2 = plt.figure(2)
    plt.plot(iterations, sgd.costs, label="Costs")
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost")
    plt.title("Stochastic Gradient Descent Cost at Each Iteration")
    plt.legend()

    plt.show()

    analytical_w = lr.calculate_analytical_weights(sgd.training_examples, sgd.training_outputs)
    print(analytical_w)
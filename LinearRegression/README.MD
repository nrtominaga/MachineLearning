# Linear Regression

Make to sure to import the `Linear_Regression` class
```python  
from Linear_Regression import Linear_Regression as lr
```

To run either of linear regression algorithms:

```python
bgd = lr(training_file, test_file, r, lr.batch_gradient_descent)
sgd = lr(training_file, test_file, r, lr.stochastic_gradient_descent)
```

where `training_file = './Data/.../train.csv'`, `test_file = ./Data/.../test.csv`, `r` is the learning rate, and the final argument is the gradient descent method you wish to choose either `lr.batch_gradient_descent` or `lr.stochastic_gradient_descent`

The `Linear_Regression` objects have a `w` field which is the learned weights and a `costs` field which is each iterations testing data cost.
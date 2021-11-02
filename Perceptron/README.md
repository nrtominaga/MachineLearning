# Perceptron

Import Perceptron
```python
from Perceptron import Perceptron
```

Then to run the perceptron algorithm simply write
```python
perceptron = Perceptron(train_file, test_file, r, T, Perceptron.perceptron)
```
where `train_file` is the location of your training file, `test_file` is our test file, `r` is our learning rate, `T` is
the number of epochs, and `Perceptron.perceptron` is the which perceptron algorithm we wish to use.  The three 
perceptron algorithms provided are `Perceptron.perceptron` which is our standard perceptron, 
`Perceptron.voted_perceptron` which is our voted perceptron, and `Perceptron.averaged_perceptron` which is our averaged 
perceptron.  To get the error simply run
```python
error = perceptron.perform_error_function()
```
and this will return the error to the variable `error`
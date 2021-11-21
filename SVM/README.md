# SVM

The primal SVM and Dual SVM algorithms have been implemented:

### Primal SVM
```python
from SVM import SVM
```

To train the primal SVM algorithms:
```python
primal_svm = SVM(training_file, testing_file, epochs, gamma, C, a=None, KfoldCV=1)
```

where `training_file` is the location of your training file, `testing_file` is our test file, `epochs` is the number of epochs, `gamma` is our learning rate, and `C` is the C value.

When `a = None`, we use the learning rate schedule:

<img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\gamma_t&space;=&space;\frac{\gamma_0}{1&plus;t}" title="\gamma_t = \frac{\gamma_0}{1+t}" />

Otherwise when `a` is defined we use the learning rate schedule: 

<img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\gamma_t&space;=&space;\frac{\gamma_0}{1&plus;\frac{\gamma_0t}{a}}" title="\gamma_t = \frac{\gamma_0}{1+\frac{\gamma_0t}{a}}" />

When `KfoldCV > 1` we will run k-fold cross validation rather than training on the full dataset therefore no weights are learned.  The average validation errors can be accessed through `primal_svm.avg_error`.

To get the error for a dataset e.g. the testing dataset you can run:
```python
testing_error = SVM.error(primal_svm.weights, primal_svm.testing_examples, primal_svm.testing_labels)
```

### Dual SVM
```python
from SVM import Dual_SVM
```

To train the dual SVM algorithms:
```python
dual_svm = Dual_SVM(training_file, testing_file, C, gamma=None)
```
where `training_file` is the location of your training file, `testing_file` is our test file, `C` is the C value and `gamma` is our learning rate for the Guassian kernel.

When `gamma = None` then the linear kernel is utilized otherwise the algorithm will use the Gaussian kernel.

To get the training/testing error:
```python
error = dual_svm.error(error_type='testing', data=None)
```
When `error_type = 'testing'/'training'` then we get our testing/training error respectively.  Otherwise we assume the `data = (examples, labels)` parameter has the examples we want to get the error. 

To run the experiments simply run the `run.sh` file.  The experiments take a long time (>1 hr) to run so the output is already stored in `output.txt` for grading convenience.
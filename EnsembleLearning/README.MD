# Ensemble Learning

Make to sure to import the `Node` class and each of the `Ensemble Learning` classes
```python  
from ID3 import Node
from Ensemble_Learning import Adaboost as ada, Bagging as bg, Random_Forest as rf 
```

The setup for attributes and training/testing examples is the same as for the `DecisionTree` module.

```python
attribute_names = ['O', 'T', 'H', 'W', 'Play']
attribute_vals = [['S', 'O', 'R'], ['H', 'M', 'C'], ['H', 'N', 'L'], ['S', 'W'], ['+', '-']]
```

Load in data from csv file using these commands:

```python
training_examples = Node.open_data('./Data/.../train.csv')  # load in training data
test_examples = Node.open_data('./Data/.../test.csv')  # load in car test data
```

To create the attributes use this command:

```python
attributes = Node.create_attributes(attributes_names, attributes_vals, training_examples,
                                       test_examples)
```

How to perform `Adaboost`, `Boosting`, and `Random Forest` respectively:

```python
adaboost = ada(training_examples, test_examples, attributes, numb_iterations)
bagging = bg(training_examples, test_examples, attributes, numb_iterations)
random_forest = rf(training_examples, test_examples, attributes, numb_iterations, s)
```
Each algorithm will have a stored list for the on the fly training and testing errors

You can run the experiment by running the `run.sh` file although the code for a lot of the experiments is commented out to help with runtime while completing this assigment.  Experiment code is located in `experiment.py`.
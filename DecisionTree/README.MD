# Decision Trees

Make to sure to import the `Node` class.
```python
from ID3 import Node
```

To learn a decision tree create an array of all the attributes with the label at the end of the array.  Then create an array of arrays where each array contains the values of each of the attributes. Make sure the attribute names and their respective values show up in the examples/csv files in the same order. If the value is numeric replace the value array with `'numeric'`.

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

Then to create a tree utilize the `Node.id3` method:

```python
tree = Node.id3(training_examples, attributes, info_gain_method, 0, max_depth)
```
`info_gain_method = Node.info_gain`, `info_gain_method = Node.majority_error`, or `info_gain_method = Node.majority_error` and `max_depth = ` however deep you want your tree to be.

If you want to check your error for either your training or test examples
```python
train_error = Node.check_error(tree, training_examples, attributes)
test_error = Node.check_error(tree, test_examples, attributes)
``` 
`Node.check_error` returns a percentage value.

If you want to use the `attributes` variable twice to make a different tree make sure to run `Node.reclaim_attributes(attributes)`
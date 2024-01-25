# MultiLayer Perceptron

A python implementation of a Multi Layer Perceptron (MLP) using numpy.

![results](images/predictions%20vs%20ground%20truth.png)

## Usage

- `run.py` - Sample usage for the Neural Network on a noisy sine wave.
- `nn.py` - Implementation of Fully Connected layer and Neural Network (NN) class

Running `run.py` will start a training on the noisy sine function.

### Sample output

While training, metrics will print on every epoch. Sample output:

```
starting training...
epoch: 10, train RMSE: 3.052E-01, val RMSE: 2.993E-01
epoch: 20, train RMSE: 2.998E-01, val RMSE: 2.970E-01
epoch: 30, train RMSE: 3.000E-01, val RMSE: 2.941E-01
...
```

Two plots will be shown after the training finishes. An error vs. epochs plot and a predictions vs. ground truth plot. Samples are included in `images/`.

## Attribution

The implemented algorithm is based on the book Tom M. Mitchell - Machine Learning. It explains neural networks and the backpropagation algorithm.

## Note

This code was developed on another repository and was then ported to this one.
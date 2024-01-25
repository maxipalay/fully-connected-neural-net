import numpy as np
import matplotlib.pyplot as plt
from nn import FCLayer, NN


def test_network():
    """ Test neural network, aim is learning the sine function. """

    # generate some data
    x = np.linspace(0, 2*np.pi, 2000)  # even distribution across x
    np.random.shuffle(x)            # shuffle x's
    y = np.sin(x)+np.random.rand(x.shape[0])/10.0  # calculate y's
    # get the final form of X,Y
    X = np.array(x/2/np.pi)
    Y = y/y.max()
    # Note we're offsetting Y's as we're using sigmoid activations,
    # which don't allow the network to predict negative values.
    validation_split = 0.15
    split_index = int(np.round(X.shape[0]*(1-validation_split)))

    X_train = X[:split_index]
    Y_train = Y[:split_index]

    X_val = X[split_index:]
    Y_val = Y[split_index:]

    # set params
    lr = 0.1
    epochs = 500

    # instance layers & network
    layers = [FCLayer("input", 1, 20),
              FCLayer("hidden_1", 20, 20),
              FCLayer("output", 20, 1)]
    nn = NN(layers)
    # train
    train_metric, validation_metric = nn.train(learning_rate=lr,
                                               max_epochs=epochs,
                                               x_train=X_train,
                                               y_train=Y_train,
                                               x_val=X_val, y_val=Y_val,
                                               batch_size=64,
                                               threshold=0.01)

    # get predictions
    preds_train = nn.predict(X_train)
    preds_val = nn.predict(X_val)

    # plot predictions vs ground truth
    plt.figure()
    plt.scatter(X, Y)
    plt.scatter(X_train, np.array(preds_train)[:, 0])
    plt.scatter(X_val, np.array(preds_val)[:, 0])
    plt.title('Neural network test on learning the sine function')
    plt.legend(['ground truth', 'predictions on training set',
                'predictions on validation set'])
    plt.figure()
    plt.plot(train_metric)
    plt.plot(validation_metric)
    plt.title('Training metrics')
    plt.legend(['RMSE on training set', 'RMSE on validation set'])
    plt.show()

    return


if __name__ == "__main__":
    test_network()

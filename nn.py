"""
Algorithm is based on Tom M. Mitchell - Machine Learning.

The book explains neural networks and the backpropagation algorithm.
"""

import numpy as np

np.random.seed(44)

class FCLayer():
    """ Represents a fully connected layer. """
    def __init__(self, name, input_size, output_size):
        self.weights = (np.random.randn(input_size, output_size))/10.0
        self.biases = np.zeros(output_size)
        self.name = name

    def forward(self, input_data):
        return self.activation(np.dot(input_data, self.weights) + self.biases)

    def activation(self, vector):
        return np.tanh(vector)

    def activation_derivative(self, vector):
        return 1.0 - np.multiply(np.tanh(vector),np.tanh(vector))

class NN():
    """ Represents a Neural network, in this case consisting of fully connected layers. """
    def __init__(self, layers):
        self._layers = layers

    def predict(self, X):
        """ Perform inference on X. """
        ys = []

        for x in X:
            inx = x
            for layer in self._layers:
                inx = layer.forward(inx)
            ys.append(inx.copy())
        return ys
    
    def metric(self, prediction, truth):
        return np.linalg.norm(prediction - truth)

    def train(self, learning_rate, max_epochs, x_train, y_train, x_val, y_val, batch_size=1, momentum=0.6, shuffle=True, threshold=0.03):
        """ Train using mini-batch gradient descent with momentum. """
        
        # we're storing metrics to return them
        mse_train = []
        mse_val = []
        
        print("starting training...")
        
        # Iterate over epochs
        for epoch in range(max_epochs):
            # check we're above the threshold
            if len(mse_val) > 0 and mse_val[-1] <= threshold:
                break
            # every 10 epochs, print metrics
            if epoch % 10 == 0 and len(mse_val)>0:
                print(f"epoch: {epoch}, train RMSE: {mse_train[-1]:.3E}, val RMSE: {mse_val[-1]:.3E}")
            
            # we store the metrics for every batch in this array
            epoch_train_mse = []
            
            # shuffle the data for each epoch
            if shuffle:
                indices = np.arange(len(x_train))

                np.random.shuffle(indices)

                x = x_train[indices]
                y = y_train[indices]
            else:
                x = x_train
                y = y_train

            # initialize velocities for momentum
            velocities_weights = [np.zeros_like(layer.weights) for layer in self._layers]
            velocities_biases = [np.zeros_like(layer.biases) for layer in self._layers]

            # Iterate over batches
            for i in range(0, len(x), batch_size):
                X_batch = x[i:i + batch_size]
                Y_batch = y[i:i + batch_size]
                
                batch_train_mse = []

                # cccumulate gradients over the batch
                accum_gradients = [np.zeros_like(layer.weights) for layer in self._layers]
                accum_biases = [np.zeros_like(layer.biases) for layer in self._layers]

                # for each sample in the batch
                for x_s, y_s in zip(X_batch, Y_batch):
                    outputs = []
                    inputs = []
                    input_data = x_s.copy()

                    # Forward pass
                    for layer in self._layers:
                        inputs.append(input_data.copy())
                        outputs.append(layer.forward(inputs[-1]))
                        input_data = outputs[-1]

                    deltas = []

                    # add the error to this batch's metrics
                    batch_train_mse.append(self.metric(outputs[-1],y_s))
                    
                    # backward pass
                    output_error = y_s - outputs[-1]
                    delta = self._layers[-1].activation_derivative(outputs[-1]) * output_error
                    deltas.append(delta.copy())

                    layer_index = len(self._layers) - 2
                    while layer_index >= 0:
                        sigma = np.dot(deltas[-1], self._layers[layer_index + 1].weights.transpose())
                        delta = self._layers[layer_index].activation_derivative(outputs[layer_index])
                        delta = np.multiply(delta,sigma)
                        deltas.append(delta.copy())
                        layer_index -= 1

                    deltas.reverse()

                    # accumulate gradients
                    for index, layer in enumerate(self._layers):
                        accum_gradients[index] += np.outer(inputs[index], deltas[index])
                        accum_biases[index] += learning_rate * deltas[index].reshape(-1)

                # update weights and biases after processing the batch with momentum
                for index, layer in enumerate(self._layers):
                    velocities_weights[index] = momentum * velocities_weights[index] + learning_rate * (accum_gradients[index] / batch_size)
                    velocities_biases[index] = momentum * velocities_biases[index] + learning_rate * (accum_biases[index] / batch_size)
                    layer.weights = layer.weights + velocities_weights[index]
                    layer.biases = layer.biases + velocities_biases[index]

                # add the metrics values for this batch
                epoch_train_mse.append(np.mean(batch_train_mse))

            # add this epoch's error for training set
            mse_train.append(np.mean(epoch_train_mse))
            # add the validation error
            y_val_pred = self.predict(x_val)
            temp = []
            for a,b in zip(y_val,y_val_pred):
                temp.append(self.metric(a,b))
            mse_val.append(np.mean(temp))

        return mse_train, mse_val
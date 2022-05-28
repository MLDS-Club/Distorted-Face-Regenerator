from random import random
from math import exp
from feature_extraction import getFeatureVector
from stylegan2 import seed2vec
from stylegan2 import init_random_state
from stylegan2 import display_image
from stylegan2 import generate_image

class Network:
    # Initialize a network
    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    # Calculate neuron activation for an input
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation
    
    # Transfer neuron activation
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))
    
    # Forward propagate input to a network output
    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)
    
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
    
    # Update network weights with error
    def update_weights(self, network, z, l_rate):
        for i in range(len(network)):
            inputs = z
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= l_rate * neuron['delta']

    # Train network
    def train_network(self, network, z, l_rate, n_epoch, expected, Gs):
        for epoch in range(n_epoch):
            sum_error = 0
            outputs = forward_propagate(network, z)
            img = generate_image(Gs, outputs, 1.0)
            feat = getFeatureVector(img)
            sum_error += sum([(expected[i]-feat[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, z, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        outputs = forward_propagate(network, z)
        img = generate_image(Gs, outputs, 1.0)
        feat = getFeatureVector(img)
        sum_error += sum([(expected[i]-feat[i])**2 for i in range(len(expected))])
        print("Outputs: ", outputs)
        display_image(img)
        
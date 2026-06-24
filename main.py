import loader
import network

training_data, validation_data, test_data = \
    loader.load_data_wrapper()

print("loaded data")

# this model has 30 neurons in the hidden layer.
model = network.Network([784, 30, 10])

print("initialised model")
print(model.sizes)

initial_weights = model.weights[0]

print(f"before training: {model.evaluate(test_data)} / {len(test_data)}")

# training over 30 epochs, with a mini-batch size of 10, and a learning rate of eta=3.0
model.SGD(training_data, 30, 10, 3.0, test_data=test_data)

final_weights = model.weights[0]

print(initial_weights == final_weights)
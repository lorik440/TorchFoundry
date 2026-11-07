# Converts images into PyTorch tensors so they can be used by the model.
from torchvision.transforms import ToTensor

# Import TorchFoundry, which provides simplified machine learning utilities.
import torchfoundry as tf

# Import PyTorch's neural network layers and PyTorch itself.
from torchfoundry import nn, torch

# Load the FashionMNIST dataset.
# The dataset contains grayscale images of clothing, each measuring 28x28 pixels.
# shuffle=True randomizes the training data order.
# root specifies where the dataset will be stored.
# ToTensor() converts each image into a tensor.
train_data, test_data = tf.loadDataset(
    "FashionMNIST",
    shuffle=True,
    root="examples/FashionMNIST_example/data",
    transformer=ToTensor()
)

# Get a batch of training images and labels to inspect their shapes.
images, labels = next(iter(train_data))
print(images.shape)

# Define the neural network architecture by listing its layers.
# Each layer processes the output of the previous layer.
layers = (
    # Flatten each 28x28 image into a one-dimensional vector of 784 values.
    nn.Flatten(),

    # Transform the 784 input values into 64 features.
    nn.Linear(28 * 28, 64),

    # Apply ReLU activation to introduce non-linearity.
    nn.ReLU(),

    # Reduce the 64 features to 16.
    nn.Linear(64, 16),

    # Apply another ReLU activation.
    nn.ReLU(),

    # Produce 10 output values, one for each FashionMNIST class.
    nn.Linear(16, 10)
)


# Create the neural network using the architecture defined above.
# path specifies where the model's weights can be loaded from or saved to,
# depending on how TorchFoundry implements createModel.
# show_Summary=True requests a summary of the model architecture.
# input_size describes the example input shape used for the summary:
# (batch_size, channels, height, width).
model = tf.createModel(
    layers,
    path="examples/FashionMNIST_example/model.pth",
    show_Summary=True,
    input_size=(1, 1, 28, 28)
)


# Configure the Adam optimizer, which updates the model's weights during training.
# lr is the learning rate, controlling the size of each update.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# Define the loss function.
# CrossEntropyLoss measures how far the model's predictions are from the correct classes.
# It is commonly used for multi-class classification.
loss_fn = nn.CrossEntropyLoss()


# Train the model using the training dataset.
# The loss function measures prediction errors, and the optimizer updates the weights.
# epoch=5 means the model will go through the training dataset five times.
model = tf.trainModel(
    model,
    train_data,
    loss_fn,
    optimizer,
    epoch=5,
    path="examples/FashionMNIST_example/model.pth",
)


# Evaluate the trained model using the test dataset.
# This helps measure how well the model performs on data used for evaluation
# rather than for updating its weights.
tf.testModel(
    model, 
    test_data, 
    loss_fn,
    path="examples/FashionMNIST_example/model.pth",
)


# Save the trained model's weights so they can be loaded and reused later.
tf.saveModel(
    model,
    path="examples/FashionMNIST_example/model.pth"
)

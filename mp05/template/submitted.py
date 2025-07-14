import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.

    """

    block = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 5)
    )

    return block
    #raise NotImplementedError("You need to write this part!")


def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """

    return torch.nn.CrossEntropyLoss()
    raise NotImplementedError("You need to write this part!")

# Code referenced from the mp5 notebook

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(hidden_size, output_size)
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x_temp = self.hidden(x)
        x_temp = self.relu(x_temp)
        y = self.output(x_temp)
        #raise NotImplementedError("You need to write this part!")
        return y
        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer

    labels = set()
    for i, label in train_dataloader:
        # Assuming labels are tensor, convert them to a list of unique labels
        labels.update(label.tolist())
    
    lrate = 0.01
    model = NeuralNet(2883, 200, len(labels))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lrate)
    

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_dataloader):

            model.train()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################

    return model



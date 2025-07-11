# Deployment of required libraries
# pip install japanize_matplotlib
# pip install torchviz
# pip install torchinfo


# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
# import japanize_matplotlib
# from IPython.display import display

# Import of torch-related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot

# Change default font size
plt.rcParams['font.size'] = 14

# Change default graph size
plt.rcParams['figure.figsize'] = (6,6)

# Dialog display ON by default
plt.rcParams['axes.grid'] = True

# Setting the number of numpy digits to display
np.set_printoptions(suppress=True, precision=4)



#----------------------------------------------------------------------------------------------------------------

# Preparation of training data
# Importing Libraries
from sklearn.datasets import load_iris

# data loading
iris = load_iris()

# Input data and correct data acquisition
x_org, y_org = iris.data, iris.target

# result check
print('original data', x_org.shape, y_org.shape)


# data refinement
# For input data, only sepal length(0) and petal length(2) are extracted
x_select = x_org[:,[0,2]]

# result check
print('original data', x_select.shape, y_org.shape)


# Split into training data and validation data (shuffle at the same time)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_select, y_org, train_size=75, test_size=75,
    random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# Split data by correct answer value
x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
x_t2 = x_train[y_train == 2]



#----------------------------------------------------------------------------------------------------------------

# Parameter setting for learning
# input
n_input = x_train.shape[1]

# output
n_output = len(list(set(y_train)))

# Check input and output dimensions
print(f'n_input: {n_input}  n_output: {n_output}')

# predictive functon
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, self.l1)
        self.l2 = nn.Linear(self.l1, n_output)

        # w and b are initially set to 1
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1

#----------------------------------------------------------------------------------------------------------------



# learning rate
lr = 0.01

# initialization
net = Net(n_input, n_output)

# loss function：cross-entropy function
criterion = nn.CrossEntropyLoss()

# optimization function: gradient descent method
optimizer = optim.SGD(net.parameters(), lr=lr)

# Number of times repeated
num_epochs = 10000

# For recording evaluation results
history = np.zeros((0,5))

# Tensor variableization of the input variable x_train and the correct answer value y_train
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).long()

# Tensor Variableization of Validation Variables
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).long()


# Prediction Calculation
outputs = net(inputs)

# Loss Calculation
loss = criterion(outputs, labels)

# Iterative calculation main loop
for epoch in range(num_epochs):

    # Training Phase
    # Initialization of gradient
    optimizer.zero_grad()

    # Prediction calculation
    outputs = net(inputs)

    # Loss calculations
    loss = criterion(outputs, labels)

    # Gradient Calculation
    loss.backward()

    # Parameter modification
    optimizer.step()

    # Predicted label calculation
    predicted = torch.max(outputs, 1)[1]

    # Loss and accuracy calculations
    train_loss = loss.item()
    train_acc = (predicted == labels).sum()  / len(labels)


    # Prediction Phase
    # Prediction calculation
    outputs_test = net(inputs_test)

    # Loss calculations
    loss_test = criterion(outputs_test, labels_test)

    # Predicted label calculation
    predicted_test = torch.max(outputs_test, 1)[1]

    # Loss and accuracy calculations
    val_loss =  loss_test.item()
    val_acc =  (predicted_test == labels_test).sum() / len(labels_test)

    if ((epoch) % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))



#----------------------------------------------------------------------------------------------------------------

# Check for loss and accuracy
print(f'initial state: loss: {history[0,3]:.5f} accuracy: {history[0,4]:.5f}' )
print(f'final state: loss: {history[-1,3]:.5f} accuracy: {history[-1,4]:.5f}' )

# Display of learning curve (loss)
plt.plot(history[:,0], history[:,1], 'b', label='training')
plt.plot(history[:,0], history[:,3], 'k', label='verification')
plt.xlabel('Number of times repeated')
plt.ylabel('loss')
plt.title('learning curve(loss)')
plt.legend()
plt.show()

# Display of learning curve (accuracy)
plt.plot(history[:,0], history[:,2], 'b', label='training')
plt.plot(history[:,0], history[:,4], 'k', label='verification')
plt.xlabel('Number of times repeated')
plt.ylabel('accuracy')
plt.title('learning curve(accuracy)')
plt.legend()
plt.show()
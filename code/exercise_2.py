import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn

from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit as it

#########################################################
################### Task 1 ##############################
#########################################################
observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

karg = {'observed': observed, 'predicted': predicted}

my_sk_mse = lambda observed, predicted: sk_mse(observed, predicted)

factory = {
    'mse_vanilla' : vanilla_mse,
    'mse_numpy' : numpy_mse,
    'mse_sk' : my_sk_mse
}

mse_list = []
for talker, worker in factory.items():
    exec_time = it.timeit(
        '{worker(**karg)}', 
        globals=globals(), 
        number=100
    ) / 100
    mse = worker(**karg)
    print(
        f"Mean Squared Error, {talker} :", mse, 
        f"Average execution time: {exec_time} seconds"
    )
    mse_list.append(mse)
    
assert(mse_list[0] == mse_list[1] == mse_list[2])
print('Test successful')



#########################################################
################### Task 2 ##############################
#########################################################
def oscillator(
        t:np.array, d:float, w0:float, 
        noise:float=0
    ) -> np.array:
    # Compute the damped frequency of the oscillator
    w = np.sqrt(w0**2 - d**2)
    
    # Compute the phase shift based on damping
    phi = np.arctan(-d / w)

    # Compute the amplitude correction factor
    A = 1 / (2 * np.cos(phi))

    # Compute the oscillatory component
    cos = np.cos(phi + w * t)

    # Compute the exponential decay component
    exp = np.exp(-d * t)

    # Compute the final solution for displacement u
    oscillations = exp * 2 * A * cos

    oscillations += np.random.rand(len(t)) * noise

    return oscillations

# Set parameter
noise = [0, .5, 1e-1]
n_points = 200
a, b = 0, 2
time = np.linspace(a, b, n_points)
range_ = b - a
d, w0 = 2, 20
oscillations = [oscillator(
    time, 
    d, w0,
    noise=noise_
) for noise_ in noise]

plt.figure()
plt.title('Oscillations')
[plt.plot(time, oscillation, label=f'Noise: {noise_}') for noise_, oscillation in zip(noise, oscillations)]
plt.grid()
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

print(f'Data generated: {n_points}, {range_}, noise level: {noise}')



#########################################################
################### Task 3 ##############################
#########################################################

X = np.array([time, oscillations[0]]).T


# Use k-means to cluster the data
n_clusters = np.linspace(2, n_points, n_points-1, dtype=int)
variance = []
for n_cluster_ in n_clusters:
    y_pred = KMeans(n_clusters=n_cluster_).fit_predict(X)

    variance.append(
        np.mean(np.array([np.var(y_pred == i)  for i in range(n_cluster_)]))
    )

print('Info about Clustering Method:\n',
    'Method: KMeans\n'
    'Parameters: \n'
    'Maximum number of iterations: 300 (sci-kit learn default\n'
    'Tolerance: 1e-4 (sci-kit learn default)\n'
)

# Plot the variance
fig, axs = plt.subplots()
axs.plot(n_clusters, variance)
axs.set_xlabel('Number of clusters')
axs.set_ylabel('Variance')
axs.grid()
fig.show()



#########################################################
################### Task 4 ##############################
#########################################################

# Number a) Linear Regression
x_regressor = X[:,0].reshape(-1, 1)
y_regressor = X[:,1].reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(x_regressor, y_regressor)

y_regression = linear_regressor.predict(x_regressor)

number_cv = 10
cv_results = cross_validate(
    linear_regressor,
    x_regressor,
    y_regressor,
    cv=number_cv,
    scoring='neg_mean_squared_error'
)
fig, axs = plt.subplots()
fig.suptitle('Linear Regression - Cross validation scores')
axs.semilogy(np.linspace(1, number_cv, number_cv), -cv_results['test_score'])
axs.grid()
axs.set_xlabel('Cross Validation')
axs.set_ylabel('MSE')
fig.show()

print('Task completed Linear Regression')

# Number b) NN (Keras)

NeuralNet = Sequential([
    Dense(256, input_shape=(1,)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=1e-3)
NeuralNet.compile(
    optimizer=optimizer, 
    loss='mean_squared_error'
)

y_nn, history = [], []
loss_mse = []
number_trainers = 20
for i in range(number_trainers):
    history.append(NeuralNet.fit(
        x_regressor, 
        y_regressor, 
        epochs=50, 
        batch_size=32,
    ))

    y_nn.append(
        NeuralNet.predict(x_regressor)
    )
    loss_mse.append(history[-1].history['loss'][-1])

fig, axs = plt.subplots()
fig.suptitle('Neural Network over Epochs')
axs.plot(x_regressor, y_regressor, label='True')
[axs.plot(x_regressor, y_nn_, label=f'Epochs = {50*((i*4)+1)}') for y_nn_, i in zip(y_nn[::4], range(int(number_trainers/4)))]
axs.legend()
axs.grid()
axs.set_xlabel('Time')
axs.set_ylabel('Amplitude')
fig.show()

error_NeuralNet = [sk_mse(y_regressor, y_nn_) for y_nn_ in y_nn]
fig, axs = plt.subplots()
fig.suptitle('Error over Epochs - Neural Net')
axs.semilogy(np.linspace(50, number_trainers*50, number_trainers), error_NeuralNet)
axs.legend()
axs.grid()
axs.set_xlabel('Epochs')
axs.set_ylabel('Error (mse)')
fig.show()

# Assume to not know the truth anymore
fig, axs = plt.subplots()
fig.suptitle('Neural Network loss function over Epochs \n We dont know the truth anymore')
axs.semilogy(np.linspace(50, number_trainers*50, number_trainers), loss_mse)
axs.grid()
axs.set_xlabel('Epochs')
axs.set_ylabel('MSE')
fig.show()

print('Task completed Neural Network')


# Number c) PINNs (PyTorch)
# Disclaimer: Most of the PINN code is just copied and pasted from a tutorial from Jassem Abbasi (jassem.abbasi@uis.no)

# Function to compute the exact solution of an underdamped harmonic oscillator
def exact_solution(d, w0, t):
    """
    Computes the exact analytical solution of an under-damped harmonic oscillator.
    
    Parameters:
        d  : float - Damping coefficient (must be smaller than natural frequency w0)
        w0 : float - Natural frequency of the system
        t  : tensor - Time values at which the solution is computed (PyTorch tensor)
    
    Returns:
        u : tensor - The displacement of the oscillator at time t
    """
    
    # Ensure that damping coefficient d is less than the natural frequency w0
    assert d < w0, "The system must be underdamped (d < w0) for oscillations to occur."

    # Compute the damped frequency of the oscillator
    w = np.sqrt(w0**2 - d**2)
    
    # Compute the phase shift based on damping
    phi = np.arctan(-d / w)

    # Compute the amplitude correction factor
    A = 1 / (2 * np.cos(phi))

    # Compute the oscillatory component
    cos = torch.cos(phi + w * t)

    # Compute the exponential decay component
    exp = torch.exp(-d * t)

    # Compute the final solution for displacement u
    u = exp * 2 * A * cos

    return u

# Custom activation function: Sine Activation
class SinActivation(nn.Module):
    """
    Defines a custom activation function using the sine function.
    This can be useful in certain types of neural networks, such as physics-informed neural networks (PINNs).
    """

    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        """
        Forward pass for the activation function.

        Parameters:
            x : tensor - Input tensor
        
        Returns:
            torch.sin(x) : tensor - Element-wise sine activation applied to x
        """
        return torch.sin(x)


# Fully Connected Neural Network (FCN)
class FCN(nn.Module):
    """
    Defines a standard fully connected (dense) neural network in PyTorch.

    This network consists of:
    - An input layer followed by an activation function
    - Multiple hidden layers with activation functions
    - An output layer without activation

    Parameters:
        N_INPUT  : int - Number of input features
        N_OUTPUT : int - Number of output features
        N_HIDDEN : int - Number of neurons in each hidden layer
        N_LAYERS : int - Number of hidden layers
        activation : nn.Module - Activation function (default: Tanh)
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation=nn.Tanh):
        super().__init__()

        # Input layer: First fully connected layer with an activation function
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),  # Linear transformation
            activation()  # Activation function (e.g., Tanh)
        )

        # Hidden layers: Stack of fully connected layers with activation functions
        self.fch = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(N_HIDDEN, N_HIDDEN),  # Fully connected layer
                    activation()  # Activation function
                ) for _ in range(N_LAYERS - 1)  # Repeat for the number of hidden layers
            ]
        )

        # Output layer: Final linear transformation (no activation function)
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Parameters:
            x : tensor - Input tensor
        
        Returns:
            x : tensor - Output tensor after passing through the network
        """
        
        x = self.fcs(x)  # Pass input through the first layer
        x = self.fch(x)  # Pass through the hidden layers
        x = self.fce(x)  # Final output layer (no activation)
        
        return x
    
torch.manual_seed(123)

# Define a Physics-Informed Neural Network (PINN) model using the previously defined FCN class
pinn = FCN(1, 1, 32, 3)  # 1 input neuron, 1 output neuron, 32 hidden units per layer, 3 hidden layers

# Define boundary points for the boundary loss
# The solution at t=0 is known and will be enforced as a constraint
t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)

# Define training points over the entire domain for the physics loss
# These points are used to enforce the differential equation constraint
t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)

# Define physical parameters for the damped harmonic oscillator equation
mu, k = 2 * d, w0**2  # Derived parameters for the equation formulation

# Generate exact solution for comparison (for visualization purposes)
t_test = torch.linspace(0, 1, 300).view(-1, 1)  # Test points in the range [0,1]
u_exact = exact_solution(d, w0, t_test)  # Compute the exact analytical solution

t_extra = torch.linspace(0, 2, 300).view(-1, 1)  # Extra test points for extrapolation
u_extra = exact_solution(d, w0, t_extra)  # Compute the exact solution for these points

# Define an optimizer (Adam) for training the neural network
## Here, we pass the network parameters to the optimizer, to make it able to modify them, in order to minimize the loss function.
optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)  # Learning rate: 0.001

# Training loop
pinn_solution = []
t_PINN = []
loss_PINN = []
for i in range(12001):  # Train for 12,000 iterations
    optimiser.zero_grad()  # Reset gradients before each optimization step
    
    # Define weighting coefficients for different loss terms
    lambda1, lambda2 = 1e-1, 1e-4  # Balance (regularize) the loss contributions 

    ### Compute boundary loss ###
    # The neural network should satisfy the initial condition u(0) = 1
    u = pinn(t_boundary)  # Compute the predicted output at the boundary point
    loss1 = (torch.squeeze(u) - 1) ** 2  # Enforce u(0) = 1

    # Compute the derivative of u with respect to time at t=0
    dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
    loss2 = (torch.squeeze(dudt) - 0) ** 2  # Enforce u'(0) = 0 (initial velocity)

    ### Compute physics loss ###
    # The neural network should satisfy the differential equation
    u = pinn(t_physics)  # Compute the network's output at the physics training points
    
    # Compute first derivative of u (du/dt)
    dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    
    # Compute second derivative of u (d²u/dt²)
    d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
    
    # Compute the residual of the governing equation: d²u/dt² + mu * du/dt + k * u = 0
    loss3 = torch.mean((d2udt2 + mu * dudt + k * u) ** 2)  # Mean squared residual

    ### Compute total loss and perform optimization step ###
    loss = loss1 + lambda1 * loss2 + lambda2 * loss3  # Weighted sum of losses
    loss.backward()  # Compute gradients
    optimiser.step()  # Update model parameters

    # Periodically visualize training progress
    if i % 4000 == 0:
        u = pinn(t_test).detach()  # Compute predicted solution on test points
        uext = pinn(t_extra).detach()  # Compute predicted solution on extrapolation points

        pinn_solution.append(u)
        t_PINN.append(t_test)
        loss_PINN.append(loss.item())

fig, axs = plt.subplots()
fig.suptitle('PINNs over Epochs')
axs.plot(x_regressor, y_regressor, label='True')
[axs.plot(t_PINN_, pinn_solution_, label=f'Epochs = {i*4000}') for i, (t_PINN_, pinn_solution_) in enumerate(zip(t_PINN, pinn_solution))]
axs.legend()
axs.grid()
axs.set_xlabel('Time')
axs.set_ylabel('Amplitude')
fig.show()

y_PINN_true = exact_solution(d, w0, t_test)
error_PINN = [sk_mse(y_PINN_true, y_nn_) for y_nn_ in pinn_solution]
fig, axs = plt.subplots()
fig.suptitle('Error over Epochs - PINN')
axs.semilogy(np.linspace(0, 4_000*3, 4), error_PINN)
axs.grid()
axs.set_xlabel('Epochs')
axs.set_ylabel('Error (mse)')
fig.show()

# Assume to not know the truth anymore
fig, axs = plt.subplots()
fig.suptitle('PINNs loss function over Epochs \n We dont know the truth anymore')
axs.semilogy(np.linspace(0, 4_000*3, 4), loss_PINN)
axs.grid()
axs.set_xlabel('x')
axs.set_ylabel('MSE')
fig.show()

print('Task completed PINN')

fig, axs = plt.subplots()
fig.suptitle('Comparison of the different methods')
axs.scatter(x_regressor, y_regressor, s=10, label='True')
axs.plot(x_regressor, y_regression, color='red', label='Linear Regression')
axs.plot(x_regressor, y_nn[-1], color='green', label='NN')
axs.plot(t_test.detach().numpy(), u.detach().numpy(), color='purple', label='PINN')
axs.set_xlabel('Time')
axs.set_ylabel('Amplitude')
axs.grid()
axs.legend()
fig.tight_layout()
fig.show()
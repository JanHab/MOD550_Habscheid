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
plt.savefig('data/Oscillations.png')
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
plt.figure()
plt.plot(n_clusters, variance)
plt.xlabel('Number of clusters')
plt.ylabel('Variance')
plt.grid()
plt.savefig('data/Variance_Clusters.png')
plt.show()



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
plt.figure()
plt.suptitle('Linear Regression - Cross validation scores')
plt.semilogy(np.linspace(1, number_cv, number_cv), -cv_results['test_score'])
plt.grid()
plt.xlabel('Cross Validation')
plt.ylabel('MSE')
plt.savefig('data/LinearRegressionCrossValidation.png')
plt.show()

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

fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
fig.suptitle('Neural Network')
axs[0].set_title('Prediction over Epochs')
axs[0].plot(x_regressor, y_regressor, label='True')
[axs[0].plot(x_regressor, y_nn_, label=f'Epochs = {50*((i*4)+1)}') for y_nn_, i in zip(y_nn[::4], range(int(number_trainers/4)))]
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')

error_NeuralNet = [sk_mse(y_regressor, y_nn_) for y_nn_ in y_nn]
axs[1].set_title('Error over Epochs')
axs[1].semilogy(np.linspace(50, number_trainers*50, number_trainers), error_NeuralNet)
axs[1].legend()
axs[1].grid()
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Error (mse)')

# Assume to not know the truth anymore
axs[2].set_title('Loss function over Epochs \n We dont know the truth anymore')
axs[2].semilogy(np.linspace(50, number_trainers*50, number_trainers), loss_mse)
axs[2].grid()
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('MSE')

fig.tight_layout()
fig.savefig('data/NeuralNetwork.png')
plt.show()

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

fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
fig.suptitle('PINNs')
axs[0].set_title('Prediction over Epochs')
axs[0].plot(x_regressor, y_regressor, label='True')
[axs[0].plot(t_PINN_, pinn_solution_, label=f'Epochs = {i*4000}') for i, (t_PINN_, pinn_solution_) in enumerate(zip(t_PINN, pinn_solution))]
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')

y_PINN_true = exact_solution(d, w0, t_test)
error_PINN = [sk_mse(y_PINN_true, y_nn_) for y_nn_ in pinn_solution]
axs[1].set_title('Error over Epochs')
axs[1].semilogy(np.linspace(0, 4_000*3, 4), error_PINN)
axs[1].grid()
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Error (mse)')

# Assume to not know the truth anymore
axs[2].set_title('Loss function over Epochs \n We dont know the truth anymore')
axs[2].semilogy(np.linspace(0, 4_000*3, 4), loss_PINN)
axs[2].grid()
axs[2].set_xlabel('x')
axs[2].set_ylabel('MSE')

fig.tight_layout()
fig.savefig('data/PINN.png')
plt.show()

print('Task completed PINN')

plt.figure()
plt.title('Comparison of the different methods')
plt.scatter(x_regressor, y_regressor, s=10, label='True')
plt.plot(x_regressor, y_regression, color='red', label='Linear Regression')
plt.plot(x_regressor, y_nn[-1], color='green', label='NN')
plt.plot(t_test.detach().numpy(), u.detach().numpy(), color='purple', label='PINN')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.savefig('data/Comparison.png')
plt.show()



#######################################################
################### Task 8 ############################
#######################################################

# GridWorld Environment
class GridWorld:
    """GridWorld environment with obstacles and a goal.
    The agent starts at the top-left corner and has to reach the bottom-right corner.
    The agent receives a reward of -1 at each step, a reward of -0.01 at each step in an obstacle, and a reward of 1 at the goal.

    Args:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.

    Attributes:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.
        obstacles (list): The list of obstacles in the grid.
        state_space (numpy.ndarray): The state space of the grid.
        state (tuple): The current state of the agent.
        goal (tuple): The goal state of the agent.

    Methods:
        generate_obstacles: Generate the obstacles in the grid.
        step: Take a step in the environment.
        reset: Reset the environment.
    """
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def step(self, action):
        """
        Take a step in the environment.
        The agent takes a step in the environment based on the action it chooses.

        Args:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left

        Returns:
            state (tuple): The new state of the agent.
            reward (float): The reward the agent receives.
            done (bool): Whether the episode is done or not.
        """
        x, y = self.state

        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        self.state = (x, y)
        if self.state in self.obstacles:
         #   self.state = (0, 0)
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        """
        Reset the environment.
        The agent is placed back at the top-left corner of the grid.

        Args:
            None

        Returns:
            state (tuple): The new state of the agent.
        """
        self.state = (0, 0)
        return self.state


# Q-Learning
class QLearning:
    """
    Q-Learning agent for the GridWorld environment.

    Args:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.

    Attributes:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.
        q_table (numpy.ndarray): The Q-table for the agent.

    Methods:
        choose_action: Choose an action for the agent to take.
        update_q_table: Update the Q-table based on the agent's experience.
        train: Train the agent in the environment.
        save_q_table: Save the Q-table to a file.
        load_q_table: Load the Q-table from a file.
    """
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
        """
        Choose an action for the agent to take.
        The agent chooses an action based on the epsilon-greedy policy.

        Args:
            state (tuple): The current state of the agent.

        Returns:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(self.q_table[state])  # exploitation

    def update_q_table(self, state, action, reward, new_state):
        """
        Update the Q-table based on the agent's experience.
        The Q-table is updated based on the Q-learning update rule.

        Args:
            state (tuple): The current state of the agent.
            action (int): The action the agent takes.
            reward (float): The reward the agent receives.
            new_state (tuple): The new state of the agent.

        Returns:
            None
        """
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        """
        Train the agent in the environment.
        The agent is trained in the environment for a number of episodes.
        The agent's experience is stored and returned.

        Args:
            None

        Returns:
            rewards (list): The rewards the agent receives at each step.
            states (list): The states the agent visits at each step.
            starts (list): The start of each new episode.
            steps_per_episode (list): The number of steps the agent takes in each episode.
        """
        rewards = []
        states = []  # Store states at each step
        starts = []  # Store the start of each new episode
        steps_per_episode = []  # Store the number of steps per episode
        steps = 0  # Initialize the step counter outside the episode loop
        episode = 0
        # For residuum
        res = []
        q_table_old = self.q_table.copy()
        while episode < self.episodes:
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)  # Store state
                steps += 1  # Increment the step counter

                # Update residuum
                res.append(self.q_table - q_table_old)
                q_table_old = self.q_table.copy()

                # Residuum reached
                if episode >= 10 and np.max(res[:-10]) < 1e-1:
                    return rewards, states, starts, steps_per_episode, episode

                if done and state == self.env.goal:  # Check if the agent has reached the goal
                    starts.append(len(states))  # Store the start of the new episode
                    rewards.append(total_reward)
                    steps_per_episode.append(steps)  # Store the number of steps for this episode
                    steps = 0  # Reset the step counter
                    episode += 1
        return rewards, states, starts, steps_per_episode, episode


learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]#[1e-1, 1e-2, 1e-3, 1e-4]
convergence = []

print('Starting Reinforcement Learning')
print('Check for number of iterations to reach convergence')
for lr in learning_rates:
    print('Learning Rate: ', lr)
    env = GridWorld(size=5, num_obstacles=5)
    agent = QLearning(env, alpha=lr, episodes=250)  # Set a max of 100 episodes
    
    rewards, states, starts, steps_per_episode, episodes = agent.train()

    convergence.append(episodes)

plt.figure()
plt.title('Convergence of Q-Learning')
plt.loglog(learning_rates, convergence, '--o', label='Convergence')
plt.axhline(y=250, color='r', linestyle='--', label='Max Iterations')
plt.legend()
plt.xlabel('Learning Rate')
plt.ylabel('Episodes to Convergence')
plt.grid()
plt.savefig('data/RL.png')
plt.show()
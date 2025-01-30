''' 
Author: Jan Habscheid
Mail: Jan.habscheid@rwth-aachen.de
'''

# imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Generate class for datastructure
class Data:
    def __init__(
            self, 
            x_lb:float, x_ub:float, 
            n_points:int, 
            intercept:float, slope:float, noise:float
        ):
        '''
        Constructor

        Parameters:
        x_lb (float): Lower bound of the x values
        x_ub (float): Upper bound of the x values
        n_points (int): Number of points
        intercept (float): Intercept of the linear function
        slope (float): Slope of the linear function
        noise (float): Noise of the data
        '''
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.n_points = n_points 

        self.intercept = intercept
        self.slope = slope
        self.noise = noise

        self.x, self.y = [], []

    def linear_function(self, x:np.array) -> np.array:
        '''
        Get linear y values based on the slope and intercept

        Parameters:
        x np.array: x values

        Returns:
        (np.array): Linear y values
        '''
        return self.intercept + self.slope * x

    def generate_data(
            self, 
            dist:str='linear'
        ) -> np.array:
        '''
        Generate data based on the distribution

        Appends x and y data to storage

        Parameters:
        dist (str): Distribution of the data
        Choose from 'linear', 'linear_noise'
        '''
        # Generate x data and linear y data
        x = np.random.rand(self.n_points) * (self.x_ub - self.x_lb) + self.x_lb
        y = self.linear_function(x)

        # Add noise
        if dist == 'linear_noise':
            y += np.random.randn(self.n_points) * self.noise

        self.x.append(x)
        self.y.append(y)

    def append_data(
            self, 
            index_1:int, index_2:int
        ):
        '''
        Appends data from two datasets (given by index) and stores them

        Parameters:
        index_1 (int): Index of the first dataset
        index_2 (int): Index of the second dataset
        '''

        # Extract data
        x_1 = self.x[index_1]
        x_2 = self.x[index_2]

        y_1 = self.y[index_1]
        y_2 = self.y[index_2]

        # Append data
        x = np.concatenate((x_1, x_2))
        y = np.concatenate((y_1, y_2))

        self.x.append(x)
        self.y.append(y)

    def scatter(
            self, 
            index:None|int=None,
        ):
        '''
        Scatter plot of the data
        
        Parameters:
        index (None|int): Index of the dataset to plot
        None: Plot all datasets
        int: Plot the dataset with the given index
        '''

        plt.figure()
        plt.title('Some random 2D data scatter plot')
        if index == None:
            for i, (x_, y_) in enumerate(zip(self.x, self.y)):
                plt.scatter(x_, y_, label=f'Datset {i}')
        else:
            plt.scatter(self.x[index], self.y[index], label=f'Datset {index}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig('data/Assignment_1_Scatter_plot.png')
        plt.savefig('data/Assignment_1_Scatter_plot.pdf')
        plt.show()

    def store_metadata(
            self, 
            filename:str
        ):
        '''
        Store the metadata in a csv file

        Parameters:
        filename (str): Filename of the csv file
        '''
        data = {
            'x_lb': [self.x_lb],
            'x_ub': [self.x_ub],
            'n_points': [self.n_points],
            'intercept': [self.intercept],
            'slope': [self.slope],
            'noise': [self.noise]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def store_data(
            self, 
            filename:str,
            index:int
        ):
        '''
        Store the metadata in a csv file

        Parameters:
        filename (str): Filename of the csv file
        index (int): Index of the dataset to store
        '''
        data = {
            'x': self.x[index],
            'y': self.y[index]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    
# Task 8
REPOSITORY_1 = (
    '1. Repository: https://github.com/lululxvi/deepxde \n'
    + 'Copyright free license \n'
    + 'Folder called: deepxde \n'
    + 'Different submodules in the folder and some direct files \n'
    + 'Docstrings at some functions, when unclear what this is for \n'
    + 'However, no general documentation \n'
    + 'No checking for bad inputs \n'
    + 'No testing \n'
    + 'Several examples, also listed in the automatically generated documentation documentation'
    + '\n\n\n'
)

REPOSITORY_2 = (
    '2. Repository: https://github.com/tataratat/splinepy \n'
    + 'Copright free license (MIT) \n'
    + 'Folder called: splinepy \n'
    + 'Different submodules and some direct files \n'
    + 'Strict docstrings in numpy docstring style \n'
    + 'Strict checking for bad inputs \n'
    + 'Clear naming convention (call variables what they are) \n'
    + 'Github pipeline for testing \n'
    + 'Online documentation \n'
    + 'Plenty examples to get familiar with the software'
    + '\n\n\n'
)

REPOSITORY_3 = (
    '3. Repository: https://github.com/numpy/numpy \n'
    + 'Copyright free license (numpy license) under some conditions \n'
    + 'Folder called: numpy \n'
    + 'Different submodules and some direct files \n'
    + 'No strict docstrings, if docstrings then in numpy docstring style \n'
    + 'However, it wrappes a C library, so the code is not that easy to understand \n'
    + 'Strict checking for bad inputs with plenty of error messages \n'
    + 'clear naming convention \n'
    + 'Pull request template \n'
    + 'Github pipeline for testing \n'
    + 'Online documentation \n'
    + 'Plenty examples to get familiar with the software \n'
)


if __name__ == '__main__':
    # Parameters
    x_lb = 0
    x_ub = 1
    n_points = 101
    intercept = 2
    slope = 5
    noise = 0.1

    # Initialize the class
    DataClass = Data(
        x_lb=x_lb,
        x_ub=x_ub,
        n_points=n_points,
        intercept=intercept,
        slope=slope,
        noise=noise
    )

    # Generate data (linear and linear with noise)
    DataClass.generate_data(dist='linear')
    DataClass.generate_data(dist='linear_noise')

    # Append data
    DataClass.append_data(0, 1)

    # # Scatter plot
    # DataClass.scatter(0)
    # DataClass.scatter(1)
    # DataClass.scatter(2)

    # Scatter plot
    DataClass.scatter()

    # Store data
    DataClass.store_metadata('data/Assignment_1_Data_MetaData.csv')
    DataClass.store_data('data/Assignment_1_Data.csv', -1)


    # Write repository data to txt file
    with open('data/Assignment_1_Repositories.txt', 'w') as file:
        file.write(REPOSITORY_1)
        file.write(REPOSITORY_2)
        file.write(REPOSITORY_3)

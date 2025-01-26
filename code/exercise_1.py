''' 
Author: Jan Habscheid
Mail: Jan.habscheid@rwth-aachen.de
'''

# imports
import numpy as np 
import matplotlib.pyplot as plt

# Generate class for datastructure
class Data:
    def __init__(
            self, 
            x_lb:float, x_ub:float, 
            n_points:int, 
            intercept:float, slope:float, noise:float
        ):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.n_points = n_points 

        self.intercept = intercept
        self.slope = slope
        self.noise = noise

        self.x, self.y = [], []

    def linear_function(self, x:np.array) -> np.array:
        return self.intercept + self.slope * x

    def generate_data(
            self, 
            dist:str='linear'
        ) -> np.array:
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

    def store_data(
            self, 
            filename:str
        ):
        np.savez(
            filename, 
            x_lb=self.x_lb, x_ub=self.x_ub, 
            n_points=self.n_points,
            intercept=self.intercept, slope=self.slope, 
            noise=self.noise,
            x=self.x, y=self.y
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

    # Scatter plot
    DataClass.scatter(0)
    DataClass.scatter(1)
    DataClass.scatter(2)

    # Scatter plot
    DataClass.scatter()

    # Store data
    DataClass.store_data('data/Assignment_1_data.npz')

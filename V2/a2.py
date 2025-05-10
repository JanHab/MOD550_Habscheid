import numpy as np
import matplotlib.pyplot as plt

class DataHandler():
    def __init__(
            self, 
            n:int
        ):
        self.n = n

    def data_generation(
            self, 
            lb, ub,
            function,
            noise:float
        ):
        x = np.random.uniform(lb, ub, n)
        f = function(x)
        f += np.random.randn(f.shape[0]) * noise

        return x, f

    def append_data(
            self, 
            x_1:np.ndarray, f_1:np.ndarray,
            x_2:np.ndarray, f_2:np.ndarray
        ):
        x = np.append(x_1, x_2)
        f = np.append(f_1, f_2)

        return x, f

    def plotting(
            self, 
            x:np.ndarray, f:np.ndarray, 
            title:str, filesave:str
        ):
        plt.figure()
        plt.scatter(x, f, color='blue')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.savefig(filesave)

    def save_data(
            self,
            filesave:str,
            x:np.ndarray, f:np.ndarray,
            noise:str|tuple
        ):
        np.savez(
            filesave,
            x=x,
            f=f,
            noise=noise,
        )



if __name__ == '__main__':
    n = 101
    lb, ub = 0., 10.
    datahandler = DataHandler(
        n=n
    )

    # Generate dataset 1
    function1 = lambda x: x ** 2 + 3
    noise1 = 5
    x_1, f_1 = datahandler.data_generation(
        lb, ub, 
        function1, noise1
    )
    datahandler.plotting(
        x_1, f_1, 'Dataset 1', 'Figures/Dataset1.pdf'
    )
    datahandler.save_data(
        'Data/Dataset1.npz',
        x_1, f_1,
        noise1
    )

    # Generate dataset 2
    function2 = lambda x: np.sin(x) + 3
    noise2 = 0.5
    x_2, f_2 = datahandler.data_generation(
        lb, ub,
        function2, noise2
    )
    datahandler.plotting(
        x_2, f_2, 'Dataset 2', 'Figures/Dataset2.pdf'
    )
    datahandler.save_data(
        'Data/Dataset2.npz',
        x_2, f_2,
        noise2
    )

    # Append datasets
    x_combined, f_combined = datahandler.append_data(
        x_1, f_1,
        x_2, f_2
    )
    datahandler.plotting(
        x_combined, f_combined, 'Dataset combined', 'Figures/Dataset_combined.pdf'
    )
    datahandler.save_data(
        'Data/Dataset_combined.npz',
        x_combined, f_combined,
        (noise1, noise2)
    )
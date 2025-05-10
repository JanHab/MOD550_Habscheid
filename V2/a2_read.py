import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from a2 import DataHandler

filename = [
    'Data/Dataset1.npz',
    'Data/Dataset2.npz',
    'Data/Dataset_combined.npz'
]
title = [
    'Dataset 1', 
    'Dataset 2',
    'Dataset combined'
]

for file, title in zip(filename, title):
    data = np.load(file)
    x = data['x']
    f = data['f']

    datahandler = DataHandler(n=1)
    datahandler.plotting(
        x, f,
        title, f'Figures/{title}_reproduced.pdf'
    )

    # Make linear regression
    lr = LinearRegression()
    lr.fit(x.reshape(-1,1), f.reshape(-1,1))
    plt.figure()
    plt.scatter(x, f, color='blue', label='Dataset')
    plt.plot(np.sort(x), lr.predict(np.sort(x).reshape(-1,1)).reshape(-1), color='red', label='Regression')
    plt.legend()
    plt.grid()
    plt.show()
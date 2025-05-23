# MOD550_Habscheid
Repository for the course: MOD550 from Jan Habscheid

## Installation

All the calculations were performed with **Python=3.12.2**.
Another Python version might also work, but was not tested.
Use the `requirements.txt` to install necessary dependencies

``` bash
pip install -r requirements.txt
```

### Installation with Conda

```bash
conda create --name MOD550 python=3.12.2 -y
conda activate MOD550
pip install -r requirements.txt
```

## Assignment 1

Execute with:

``` bash
python code/exercise_1.py
```

### How to use my data?

In `data/Assignment_1_Data_MetaData.csv` you find the "meta" data to reproduce my data. So with this is meant, the boundaries for the x-domain, number of points, etc. 
You can rerun the code with these parameters to get a similar output, up to some uncertainty

In `data/Assignment_1_Data.csv` on the other hand, you find the "real" data, from the appended vectors.
You can load it with pandas and visualize with matplotlib:

``` python
# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataframe
csv_file = 'data/Assignment_1_Data.csv'
df = pd.read_csv(csv_file)

# scatter plot
plt.scatter(df['x'], df['y'])
```

### Which data did I import?

Historical temperature in Oslo (last 2 years) with the additional package `meteostat`. Install it with:

``` bash
pip install meteostat
```

The truth about this data is the average temperature on each day for the last 2 years at the specified spatial position on earth.

## Assignment 2

Use `scikit-learn`for linear regression, `tensorflow` with the keras frontend for NN and `torch`for PINNs.

### Code Execution

The code execution may take some time initially to load all the packages.
Wait at least 30 seconds before interrupting the execution.

Run

``` bash
python code/exercise_2.py
```

## Contact

### Author

- Jan Habscheid
- [Jan.Habscheid@rwth-aachen.de](mailto:Jan.Habscheid@rwth-aachen.de)

### Supervisor

- Enrico Riccardi

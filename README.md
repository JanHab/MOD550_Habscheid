# MOD550_Habscheid
Repository for the course: MOD550 from Jan Habscheid

## Assignment 1

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

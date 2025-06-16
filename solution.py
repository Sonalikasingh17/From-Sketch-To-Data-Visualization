# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix

# Load csv file into pandas Data Frame
df=pd.read_csv("Pigeon.csv")
df

# Data Cleaning
df.columns=['x','y']
sns.scatterplot(data=df, x='y',y='x')

df.describe()

df.head(150)

df.tail(150)

df.info()

df['x'] = df['x'].round(2)
df['y'] = df['y'].round(2)

pd.Series(df['x'].unique())

df['x'].value_counts()
df['y'].value_counts()

df= df.drop_duplicates(subset='x', keep='first')
# Cleaned dataframes
df


sparse_matrix = csr_matrix(df)
print(sparse_matrix)
sparse_matrix.ndim
print(sparse_matrix.dtype)
print(type(sparse_matrix))


array= sparse_matrix.toarray()
print(array)
print(array.ndim)
print(array.shape)
print(type(array))

array = sparse_matrix.toarray()
print(array)
print(array.ndim)
print(array.shape)
print(type(array))

mat = 1000

min_x = np.min(array[:, 0])
min_y = np.min(array[:, 1])
max_x = np.max(array[:, 0])
max_y = np.max(array[:, 1])

# we defined min and max as it is need in discretization.

#IMPORTANT as formula i.e[ normalization * (n-1)]
discretized_x = np.clip(((array[:, 0] - min_x) / (max_x - min_x) * (mat - 1)).astype(int), 0, mat - 1)
discretized_y = np.clip(((array[:, 1] - min_y) / (max_y - min_y) * (mat - 1)).astype(int), 0, mat - 1)

print(discretized_x.ndim)  
print(discretized_x.size)  

bool_m = np.zeros((mat, mat), dtype=bool)  # boolean given.

for x, y in zip(discretized_x, discretized_y): 
    bool_m[x, y] = True   
        
sparse_matrix1 = csr_matrix(bool_m)
print(sparse_matrix1)
print(sparse_matrix1.ndim)
print(type(sparse_matrix1))

num_1 = sparse_matrix1.toarray()
print(num_1)
print(num_1.ndim)
print(num_1.size)

# Rotate by 90 degree
r_df1 = np.rot90(num_1, k=-1)
r_df1

r_df2 = np.rot90(num_1, k= -2)
r_df2


# 2nd Image
rows, cols = np.nonzero(r_df1)

# Create a subplot and scatter plot
plt.figure(figsize=(8, 8))
ax1 = plt.subplot()
ax1.scatter(rows, cols , s=40)
plt.grid(True)

# Display the plot
plt.show()

# 3rd Image
rows, cols = np.nonzero(r_df2)

# Create a subplot and scatter plot
plt.figure(figsize=(8, 8))
ax1 = plt.subplot()
ax1.scatter(rows, cols , s=40)
plt.grid(True)

# Display the plot
plt.show()

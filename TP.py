import pandas as pd
import numpy as np
from numpy.linalg import norm as l2norm


arr = [26.7, 25.8, 23.4, 74.1]

diff_list = [abs(i - j) for i in arr for j in arr]

data = pd.DataFrame(np.array(diff_list).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
print(data)


# print(pd.crosstab(df['a'], df['b']).iloc[1:, :])

arr = [np.random.rand(4) for _ in range(4)]
labels = ['1', '2', '3', '4']

for i, a in enumerate(arr):
    norm = l2norm(a)
    arr[i] = a / (norm + 1e-10)

dot_list = [np.dot(i, j) for i in arr for j in arr]

frame = pd.DataFrame(np.array(dot_list).reshape(len(arr), len(arr)),
                     index=labels, columns=labels)

print(frame)
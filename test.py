import pandas as pd
import numpy as np
# create an Empty DataFrame object
df = pd.DataFrame()

# print(df)
# df[1] = 2
# print(df)
total_dict = {}
for i in range(5):
    total_dict[i] = {}
    for j in range(5):
        total_dict[i][j] = np.random.rand()

print(total_dict)

df = pd.DataFrame.from_dict(total_dict)
print('hi')
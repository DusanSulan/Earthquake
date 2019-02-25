import matplotlib.pyplot as plt
import dask.dataframe as bag
import pandas as pd
import numpy as np
import dask.bag as db
from dask.diagnostics import ProgressBar
from pathlib import Path
#pbar = ProgressBar() # DASK PREPROCESS
#pbar.register()
#df = bag.read_csv('D:\TeorieInformaceProjekt\EarthQuake/train.csv')
#df = df.iloc[::20, :].compute()
datarows = 100000000
samplerows = 20
iter_csv = pd.read_csv('D:\TeorieInformaceProjekt\EarthQuake/train.csv', iterator=True, chunksize=10000000)
df = pd.concat([chunk[chunk.index % 20 == 0] for chunk in iter_csv])

def insert_row(idx, df, df_insert):
    return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)

dtypes={'acoustic_data': np.int16, 'time_to_failure': np.float32}
#df1 = pd.read_csv('D:\TeorieInformaceProjekt\EarthQuake/train.csv', nrows=datarows, dtype=dtypes)
#df1 = df1[df1.index % samplerows == 0]
#for i in range(2, 3, 1):#len(df) // chunkSize + 1
'''
i = 2
print(i*datarows)
df2 = pd.read_csv('D:\TeorieInformaceProjekt\EarthQuake/train.csv', skiprows=range(1, i*datarows), nrows=datarows, dtype=dtypes)
df2 = df2[df2.index % samplerows == 0]
if i == 1:
    df2.to_csv('D:/RelevantEarthQuakeBig.csv', index=True, index_label='id', header = True)
else:
    df2.to_csv('D:/RelevantEarthQuakeBig.csv', index=True, mode = 'a', index_label='id', header = False)    
#    df1 = insert_row(i*datarows, df1, df2)
df2 = pd.read_csv('D:/RelevantEarthQuakeBig.csv', dtype=dtypes)'''
plt.plot(df.index,  df['time_to_failure'] )
df['test'] = 0
#pbar = ProgressBar() # DASK PREPROCESS
#pbar.register()
#df = bag.read_csv('D:\TeorieInformaceProjekt\EarthQuake/train.csv')
#df1 = df.loc[1:40000].compute()
#df = df1.append(df2)
#df['test'] = 0
count = 1
pathlist = Path(r'D:\TeorieInformaceProjekt\EarthQuake\train').glob('**/*.csv')
for path in pathlist:
    path_in_str = str(path)
    filepath = path_in_str
    df1 = pd.read_csv(filepath)
    df1 = df1[df1.index % samplerows == 0]
    print(count)
    df1['test'] = count
    count = count + 1
    df = df.append(df1).reset_index(drop = True)

df.to_csv('D:/RelevantEarthQuake.csv', index=True, index_label='id')

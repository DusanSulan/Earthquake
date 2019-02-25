import pandas as pd
import numpy as np
import featuretools as ft
from featuretools import variable_types as vtypes
import UtilsEQ
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=190, facecolor='w', edgecolor='k')
pd.set_option('display.max_columns', None)
from pathlib import Path
from scipy.signal import argrelextrema

ft.__version__
df = pd.read_csv('D:\RelevantEarthQuake.csv',dtype={'test': np.int16,'acoustic_data': np.int16, 'time_to_failure': np.float64})#, parse_dates=['time_to_failure'])
plt.plot(df.index,  df['test'] )
plt.show()
#for index, row in dftemp.iterrows(): 
df["test"] = df["test"].astype(int)
#plt.scatter(df['time_to_failure'].values, df['acoustic_data'], s=10, alpha=0.5)
#plt.xlim(0, 2)
#plt.show()
#df = df.groupby(np.arange(len(df.index))//10, axis=0).mean()
#plt.xticks( df['time_to_failure'], df.index.values ) # location, labels

# separates the whole feature matrix into train data feature matrix, train data labels, and test data feature matrix        
feature_matrix, labels = UtilsEQ.create_features(df, defrows = 7500) #X_train, labels = UtilsEQ.get_train(feature_matrix)

from feature_selector import FeatureSelector
fs = FeatureSelector(data = feature_matrix, labels = labels) #ALL DATA HERE
fs.identify_collinear(correlation_threshold = 0.96)
fs.plot_collinear()

labels = labels[feature_matrix['test']==0]#Only train labels
#labels = np.log(labels + 1)
X_train = feature_matrix[feature_matrix['test']==0]
X_train = X_train.drop('test',1)
model = UtilsEQ.train_xgb(X_train, labels)

X_testlist = UtilsEQ.get_test(feature_matrix)

submission = pd.DataFrame()
count = 0
for X_test in X_testlist:
    submission_temp = UtilsEQ.predict_xgb(model, X_test)
    submission_temp['segment'] = count
    count = count + 1
    submission = submission.append(submission_temp)

pathlist = Path(r'D:\TeorieInformaceProjekt\EarthQuake\train').glob('**/*.csv')
submission['seg_id'] =  'o'
submission = submission.reset_index(drop = True)
count = 0
for path in pathlist:
    submission.loc[count,'seg_id'] =  str(path)
    count = count + 1
submission.to_csv('D:\SUBMISSION_EARTHQUAKE.csv' , index=True, index_label='id')




submissionTrain = UtilsEQ.predict_xgb(model, X_train)
X_train['time_to_failure'] = labels['time_to_failure']
X_train.to_csv('D:\SUBMISSION_EARTHQUAKETrain.csv' , index=True, index_label='id')
    
feature_names = X_train.columns.values
ft_importances = UtilsEQ.feature_importances(model, feature_names)
ft_importances
''' 
[30]    train-mae:0.785951      valid-mae:1.67165

Best mean score: 1.6717,
{'eta': 0.4, 'lambda': 52, 'max_depth': 8, 'feature_fraction': 0.98, 'gamma': 0.9, 'min_child_weight': 0.1, 'subsample': 0.6, 'objective': 'reg:linear', 'booster': 'gbtree', 'eval_metric': 'mae'}
Traceback (most recent call last):
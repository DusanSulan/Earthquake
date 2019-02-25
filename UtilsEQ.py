# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:47:47 2019
@author: Dusan Sulan
"""
import random
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
#from feature_selector import FeatureSelector
#for chunk in pd.read_csv(<filepath>, chunksize=<your_chunksize_here>)
#    do_processing()
#    train_algorithm()

def train_xgb(X_train, labels):
    Xtr, Xv, ytr, yv = train_test_split(X_train.values,
                                        labels,
                                        test_size=0.01,
                                        random_state=0)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)

    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    
    fixed_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'eval_metric': 'mae',}
    param_grid = {
        'eta': [0.2, 0.08, 0.4, 0.3, 0.6],
        'lambda': [26, 32, 46, 52, 58, 68],
        'max_depth': [ 6, 8, 14],
        'feature_fraction': [0.8, 0.85, 0.9, 0.95, 0.98, 1],
        'gamma': [0.8, 0.85, 0.9, 0.95, 0.98, 1, 1.1, 2, 3],
        'min_child_weight': [0, 0.05, 0.1, 0.2],# 0.4, 0.6, 0.9],
        'subsample': [0, 0.1, 0.2, 0.4, 0.6, 0.9, 0.95],}

    best_scoreNow = 999   
    for i in tqdm(range(400)):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        params.update(fixed_params)
        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=3000,
                      evals=evals, early_stopping_rounds=100, maximize=False,
                      verbose_eval=10)
        if model.best_score < best_scoreNow:
            best_scoreNow = model.best_score
            best_params = params
#    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=3000,
#                      evals=evals, early_stopping_rounds=100, maximize=False,
#                      verbose_eval=10)
    model = xgb.train(params=best_params, dtrain=dtrain, num_boost_round=3000,
                      evals=evals, early_stopping_rounds=100, maximize=False,
                      verbose_eval=10)      
    print("Best mean score: {:.4f},".format(best_scoreNow))
    print(best_params)
#    print('Modeling MAE %.5f' % model.best_score)
    return model

def predict_xgb(model, X_test):
    dtest = xgb.DMatrix(X_test.values)
    ytest = model.predict(dtest)
    X_test['XGBT'] = ytest  #np.exp(ytest) - 1 #LOGARITMUS BACK 
    return X_test[['XGBT']]

def feature_importances(model, feature_names):
    feature_importance_dict = model.get_fscore()
    fs = ['f%i' % i for i in range(len(feature_names))]
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                       'importance': list(feature_importance_dict.values())})
    f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
    feature_importance = pd.merge(f1, f2, how='right', on='f')
    feature_importance = feature_importance.fillna(0)
    return feature_importance[['feature_name', 'importance']].sort_values(by='importance',
                                                                          ascending=False)
def get_train(feature_matrix):
    X_train = feature_matrix[feature_matrix['test'] == 0]# put off test data > 0 
    X_train = X_train.drop(['test'], axis=1) # odstran sloupec(axis = 1) test je nererevantny pro trening
    labels = X_train['time_to_failure']  # LABELS = QTY
    X_train = X_train.drop(['time_to_failure'], axis=1) # odstran sloupec(axis = 1) order QTY z TRENINGU
    return (X_train, labels)

def get_test(feature_matrix):
    X_testTemp = feature_matrix[feature_matrix['test'] > 0]
    X_testlist = list()
    for i in range(0,int(X_testTemp['test'].max())):
        X_test = X_testTemp[X_testTemp['test'] == i+1]
        X_test = X_test.drop(['test'], axis=1)
        X_testlist.append(X_test)
       
    return (X_testlist)


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def roll(df, w):
    roll_array = np.dstack([df.values[i:i+w, :] for i in range(len(df.index) - w + 1)]).T
    panel = pd.Panel(roll_array, 
                     items=df.index[w-1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(w), name='roll'))
    return panel.to_frame().unstack().T.groupby(level=0)

def roll2(arr,num):
    arr=np.roll(arr,num)
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),np.nan)
    elif num > 0:
         np.put(arr,range(num),np.nan)
    return arr
def classic_sta_lta(x, length_sta, length_lta):
    
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta
# Create a training file with simple derived features
    
# Try to make more different groups
def create_features(df, defrows):
    dfmin = df['time_to_failure'][(df['time_to_failure'].shift(1) > df['time_to_failure']) & (df['time_to_failure'].shift(-1) > df['time_to_failure'])]
    lastdfmin = 0 
    lastEQrow = 1
    startendrange = pd.DataFrame(index=np.arange(20000), columns=np.arange(3))
    segmentcount = 0
    for dfmini in dfmin.index:#at the first create array of segments starts and ends and count segments
        rows = int((dfmini - lastdfmin) /round((dfmini - lastdfmin)/defrows)) #calculate rows for feature btw EQ
        segments = int((np.floor(df[df.index <dfmini].shape[0] -lastdfmin)  / rows))#find number of segments btw EQ
        lastdfmin = dfmini
        for segment in tqdm(range(segments)):
            startendrange.iloc[segmentcount,0] = lastEQrow + 1
            startendrange.iloc[segmentcount,1] = rows + lastEQrow
            startendrange.iloc[segmentcount,2] = rows
            segmentcount = segmentcount + 1
            lastEQrow = rows + lastEQrow
    #TEST SEGMENTS - different groups
    testsegments =  int(df['test'][df['test'] > 0].count() / defrows) #need int for LOOP
    lastEQrow = df.index[df['test'] > 0].min()-1
    for segment in tqdm(range(testsegments)):
        startendrange.iloc[segmentcount,0] = lastEQrow + 1
        startendrange.iloc[segmentcount,1] = defrows + lastEQrow
        segmentcount = segmentcount + 1
        lastEQrow = defrows + lastEQrow           
            
    feature_matrix = pd.DataFrame(index=range(segments), dtype=np.float64,
                           columns=['avg', 'stdv', 'max', 'min','q99', 
                                    'test'])
    labels = pd.DataFrame(index=range(segments), dtype=np.float64,
                           columns=['time_to_failure'])
    
    for segment in tqdm(range(segmentcount)):# segments = 340, range(340 je [pole 340 prvkov])
        seg = df.iloc[startendrange.iloc[segment,0]:startendrange.iloc[segment,1]]
        x = seg['acoustic_data']
        
        y = seg['time_to_failure'].values[-1]# HERE I SHOULD PUT MIN IN SEGMENT
        
        labels.loc[segment, 'time_to_failure'] = y
        feature_matrix.loc[segment, 'test'] = seg['test'].values[-1]
        feature_matrix.loc[segment, 'avg'] = x.mean()
        feature_matrix.loc[segment, 'stdv'] = x.std()
        feature_matrix.loc[segment, 'max'] = x.max()
        feature_matrix.loc[segment, 'min'] = x.min()
        feature_matrix.loc[segment, 'total_abs_sum'] = np.abs(x).sum()
        feature_matrix.loc[segment, 'q99'] = np.quantile(x,0.99)

        feature_matrix.loc[segment, 'std_first_50000'] = x[:2000].std()
        feature_matrix.loc[segment, 'std_last_50000'] = x[-2000:].std()
        feature_matrix.loc[segment, 'std_first_10000'] = x[:500].std()
        feature_matrix.loc[segment, 'std_last_10000'] = x[-500:].std()
        
        feature_matrix.loc[segment, 'avg_first_50000'] = x[:2000].mean()
        feature_matrix.loc[segment, 'avg_last_50000'] = x[-2000:].mean()
        feature_matrix.loc[segment, 'avg_first_10000'] = x[:500].mean()
        feature_matrix.loc[segment, 'avg_last_10000'] = x[-500:].mean()
        
        feature_matrix.loc[segment, 'min_first_50000'] = x[:2000].min()
        feature_matrix.loc[segment, 'min_last_50000'] = x[-2000:].min()
        feature_matrix.loc[segment, 'min_first_10000'] = x[:500].min()
        feature_matrix.loc[segment, 'min_last_10000'] = x[-500:].min()
        
        feature_matrix.loc[segment, 'max_first_50000'] = x[:2000].max()
        feature_matrix.loc[segment, 'max_last_50000'] = x[-2000:].max()
        feature_matrix.loc[segment, 'max_first_10000'] = x[:500].max()
        feature_matrix.loc[segment, 'max_last_10000'] = x[-500:].max()
        
        feature_matrix.loc[segment, 'abs_max'] = np.abs(x).max()
        feature_matrix.loc[segment, 'abs_min'] = np.abs(x).min()
        
        feature_matrix.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())
        feature_matrix.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())
        feature_matrix.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])
        feature_matrix.loc[segment, 'sum'] = x.sum()
     
        feature_matrix.loc[segment, 'q95'] = np.quantile(x, 0.95)
        feature_matrix.loc[segment, 'q99'] = np.quantile(x, 0.99)
        feature_matrix.loc[segment, 'q05'] = np.quantile(x, 0.05)
        feature_matrix.loc[segment, 'q01'] = np.quantile(x, 0.01)
        
        feature_matrix.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
        feature_matrix.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
        feature_matrix.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
        feature_matrix.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)
        
        feature_matrix.loc[segment, 'trend'] = add_trend_feature(x)
        feature_matrix.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
        feature_matrix.loc[segment, 'abs_mean'] = np.abs(x).mean()
        feature_matrix.loc[segment, 'abs_std'] = np.abs(x).std()
        
        feature_matrix.loc[segment, 'mad'] = x.mad()
        feature_matrix.loc[segment, 'kurt'] = x.kurtosis()
        feature_matrix.loc[segment, 'skew'] = x.skew()
        feature_matrix.loc[segment, 'med'] = x.median()
        
#        feature_matrix.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
#        feature_matrix.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

        feature_matrix.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        feature_matrix.loc[segment, 'q999'] = np.quantile(x,0.999)
        feature_matrix.loc[segment, 'q001'] = np.quantile(x,0.001)
#        feature_matrix.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)
      
        feature_matrix.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 20, 500).mean()
        feature_matrix.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 200, 5000).mean()
        feature_matrix.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 100, 200).mean()
        feature_matrix.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 200, 500).mean()
        feature_matrix.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 5, 50).mean()
        feature_matrix.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 5, 100).mean()
        feature_matrix.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 15, 30).mean()
        feature_matrix.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 20, 500).mean()
        feature_matrix.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=35).mean().mean(skipna=True)
        ewma = pd.Series.ewm
        feature_matrix.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=15).mean()).mean(skipna=True)
        feature_matrix.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=150).mean().mean(skipna=True)
#        feature_matrix.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=300).mean().mean(skipna=True)
        
        no_of_std = 3
        feature_matrix.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=35).std().mean()
        feature_matrix.loc[segment,'MA_700MA_BB_high_mean'] = (feature_matrix.loc[segment, 'Moving_average_700_mean'] + no_of_std * feature_matrix.loc[segment, 'MA_700MA_std_mean']).mean()
        feature_matrix.loc[segment,'MA_700MA_BB_low_mean'] = (feature_matrix.loc[segment, 'Moving_average_700_mean'] - no_of_std * feature_matrix.loc[segment, 'MA_700MA_std_mean']).mean()
        feature_matrix.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=20).std().mean()
        feature_matrix.loc[segment,'MA_400MA_BB_high_mean'] = (feature_matrix.loc[segment, 'Moving_average_700_mean'] + no_of_std * feature_matrix.loc[segment, 'MA_400MA_std_mean']).mean()
        feature_matrix.loc[segment,'MA_400MA_BB_low_mean'] = (feature_matrix.loc[segment, 'Moving_average_700_mean'] - no_of_std * feature_matrix.loc[segment, 'MA_400MA_std_mean']).mean()
        feature_matrix.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=50).std().mean()
        feature_matrix.drop('Moving_average_700_mean', axis=1, inplace=True)
        
#        feature_matrix.loc[segment, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:2000]) / x[:2500][:-1]))[0])
##        feature_matrix.loc[segment, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-2000:]) / x[-2500:][:-1]))[0])
##        feature_matrix.loc[segment, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:400]) / x[:500][:-1]))[0])
#        feature_matrix.loc[segment, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-400:]) / x[-500:][:-1]))[0])

        for w in [40 , 200, 800, 2000]:
            x_roll_std = x.rolling(w).std().dropna().values#
            x_roll_mean = x.rolling(w).mean().dropna().values#
#            x_roll_abs_mean = x_roll.mean().dropna().values
            x_roll_std = x.rolling(w).std().dropna().values#
            x_roll_mean = x.rolling(w).mean().dropna().values#
#            x_roll_abs_mean = x_roll.mean().dropna().values
            
            feature_matrix.loc[segment, 'ave_roll_std_' + str(w)] = x_roll_std.mean()
            feature_matrix.loc[segment, 'std_roll_std_' + str(w)] = x_roll_std.std()
            feature_matrix.loc[segment, 'max_roll_std_' + str(w)] = x_roll_std.max()
            feature_matrix.loc[segment, 'min_roll_std_' + str(w)] = x_roll_std.min()
            
#            feature_matrix.loc[segment, 'ave_roll_mean_' + str(w)] = x_roll_mean.mean()
#            feature_matrix.loc[segment, 'std_roll_mean_' + str(w)] = x_roll_mean.std()
#            feature_matrix.loc[segment, 'max_roll_mean_' + str(w)] = x_roll_mean.max()
#            feature_matrix.loc[segment, 'min_roll_mean_' + str(w)] = x_roll_mean.min()

            feature_matrix.loc[segment, 'q01_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.01)
            feature_matrix.loc[segment, 'q05_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.05)
#            feature_matrix.loc[segment, 'q95_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.95)
            feature_matrix.loc[segment, 'q99_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.99)
            feature_matrix.loc[segment, 'av_change_abs_roll_std_' + str(w)] = np.mean(np.diff(x_roll_std))
            feature_matrix.loc[segment, 'av_change_rate_roll_std_' + str(w)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            feature_matrix.loc[segment, 'abs_max_roll_std_' + str(w)] = np.abs(x_roll_std).max()
            

            feature_matrix.loc[segment, 'q01_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.01)
            feature_matrix.loc[segment, 'q05_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.05)
#            feature_matrix.loc[segment, 'q95_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.95)
            feature_matrix.loc[segment, 'q99_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.99)
            feature_matrix.loc[segment, 'av_change_abs_roll_mean_' + str(w)] = np.mean(np.diff(x_roll_mean))
            feature_matrix.loc[segment, 'av_change_rate_roll_mean_' + str(w)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            feature_matrix.loc[segment, 'abs_max_roll_mean_' + str(w)] = np.abs(x_roll_mean).max()

        
        
    return feature_matrix, labels

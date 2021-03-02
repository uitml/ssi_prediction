import os,sys,copy,string,random,time
import numpy as np
import pandas as pd
#import argparse

from blood_samples import nominal_values

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from multiprocessing import Pool


from scipy.stats import rankdata 

import warnings
warnings.filterwarnings("ignore")

from utils import *

TPS = time.time()

#if SAVE:
if not os.path.isdir('save'):
    os.makedirs('save')
    
lNAME = 'save/save_gboost_' + \
    PREPROC + \
    ( ('_t='+ str(TRUNC)) if not PRIMOZ else '' )+ \
    ('_log' if LOG else '') + \
    ('_ovs' if OVER_SAMPLE else '') + \
    ('_std' if STANDARDIZE else '') + \
    '.npy'
print('*** I will load: ',lNAME)
    
if SAVE:
    sNAME = 'save/ablation_save_gboost_' + \
        PREPROC + \
        ( ('_t='+ str(TRUNC)) if not PRIMOZ else '' )+ \
        ('_log' if LOG else '') + \
        ('_ovs' if OVER_SAMPLE else '') + \
        ('_std' if STANDARDIZE else '') + \
        '_ssi='+str(ABLATION) + \
        '.npy'
    print('*** I will save in: ',sNAME)


_MODEL = GradientBoostingClassifier(
            loss='deviance',
            learning_rate=.05, 
            n_estimators=100, 
            max_depth=50, 
#            subsample=1.,
            criterion='mse',
            max_features='sqrt',
#            min_samples_split=50,
#            min_samples_leaf=5,
        )
    
################################
# LOAD DATASETS

    
if PRIMOZ:
    LOAD = np.load('data/Data_Primoz.npz', allow_pickle=True)
else:
    LOAD = np.load('data/Data_Weekly.npz', allow_pickle=True)

org_values = np.vstack([ LOAD['x'], LOAD['tx'] ])
org_info = np.vstack([ LOAD['i'], LOAD['ti'] ])

del LOAD

if not PRIMOZ:
    org_values = org_values[:,:,:TRUNC]
    
    keep   = ( np.isfinite(org_values).any(1).sum(1) >= MIN_WEEK_VALUES ) * ( org_info[:,2] != 1-GENDER )
    org_values = org_values[keep]
    org_info   = org_info[keep]
    org_values = np.vstack([ nominal(org_values[i], int(org_info[i,2]), int(org_info[i,3]), LOG ) 
                    for i in range(len(org_values)) ])
    N = org_values.shape[0]
    org_avg = (
            np.nanmean( org_values.reshape( ( N, -1) )[ org_info[:,2] == 0 ] ,0),
            np.nanmean( org_values.reshape( ( N, -1) )[ org_info[:,2] == 1 ] ,0),
            )
else:
    keep = ( org_info[:,2] != 1-GENDER )
    org_values = org_values[keep]
    org_info   = org_info[keep]
    
if AUGMENT:
    LOAD = np.load('data/Data_Daily.npz', allow_pickle=True)
    aug_values = np.vstack([ LOAD['x'], LOAD['tx'] ])[:,:,:STEP[-1]]
    aug_info = np.vstack([ LOAD['i'], LOAD['ti'] ])
    del LOAD

    keep = ( np.isfinite(aug_values).any(1).sum(1) >= MIN_WEEK_VALUES ) * ( aug_info[:,2] != 1-GENDER )
    aug_values = aug_values[keep]
    aug_info = aug_info[keep]

    N,P,T = aug_values.shape

    W = len(STEP) - 1
    # Fill the missing values of test-days using the average of the week.
    # Can be nan if a variable has not been tested that week
    # If a week had no test, add a dummy results to mark the week.
    for i in range(N):
        for s0,s1 in zip(STEP[:-1],STEP[1:]):
            if np.isnan(aug_values[i,:,s0:s1]).all():
                aug_values[i,:,s0] = -1
            else:
                m = np.nanmean(aug_values[i,:,s0:s1],1)
                for s in np.isfinite( aug_values[i,:,s0:s1] ).any(0).nonzero()[0]:
                    aug_values[i,:,s0+s] = (np.isnan(aug_values[i,:,s0+s])*m) + np.nan_to_num(aug_values[i,:,s0+s])
                    
# LOAD WEIGHTS AND SORT THEM

LOAD = np.load(lNAME,allow_pickle=True).item()

if ABLATION == 'bin':
    rCOEF = []
    for wgt in LOAD['bin']['c']:
        rCOEF.append( rankdata( -np.abs(wgt) ) )
    COEF = np.argsort( np.mean( rCOEF, 0 ) )
elif ABLATION == 'ssi':
    rCOEF = []
    for wgt in LOAD['ssi']['c']:
        rCOEF.append( rankdata( -np.abs(wgt) ) )
    COEF = np.argsort( np.mean( rCOEF, 0 ) )
else:
    SSI = int(ABLATION[1])
    rCOEF = []
    for wgt in LOAD['multi']['c']:
        rCOEF.append( rankdata( -np.abs(wgt[SSI]) ) )
    COEF = np.argsort( np.mean( rCOEF, 0 ) )
    
nCOEF = len(COEF)

RESULTS = {}
RESULTS['v'] = np.zeros( (ITER, nCOEF, tCV, vCV, len(ALPHA_RANGE), 3) )
RESULTS['t'] = np.zeros( (nCOEF, ITER, tCV, 3) )
RESULTS['l'] = []
RESULTS['y'] = [] # np.zeros( (ITER, tCV, nCOEF, -1) )

RESULTS['av'] = np.zeros( (ITER, nCOEF, tCV, vCV, len(ALPHA_RANGE), 3) )
RESULTS['at'] = np.zeros( (nCOEF, ITER, tCV, 3) )
RESULTS['al'] = []
RESULTS['ay'] = [] # np.zeros( (ITER, tCV, nCOEF, -1) )


for _iter in range(ITER):

    RESULTS['l'].append([])
    RESULTS['al'].append([])
    
    RESULTS['y'].append([])
    RESULTS['ay'].append([])
    
    ################################
    # CROSS VALIDATION
    tFOLD = StratifiedKFold(n_splits=tCV, shuffle=True).split( range(org_info.shape[0]), org_info[:,0] )
    for fold_test, (idx_train_valid, idx_test) in enumerate(tFOLD):
    
        RESULTS['l'][-1].append([])
        RESULTS['al'][-1].append([])
        
        RESULTS['y'][-1].append([])
        RESULTS['ay'][-1].append([])

        ################################
        # PREPARE TRAIN/TEST SETS

        # Split train/test

        test_X, test_info, test_y = org_values[idx_test], org_info[idx_test], org_info[idx_test, 0]
        if not AUGMENT:
            if OVER_SAMPLE:
                y = org_info[idx_train_valid, 0]
                ovs = RandomOverSampler() if PRIMOZ else SMOTE()
                idx_train_valid = ovs.fit_resample(idx_train_valid[:,None], y)[0].squeeze()
            train_X, train_info, train_y = org_values[idx_train_valid], org_info[idx_train_valid], org_info[idx_train_valid, 0]
            
        else:
            train_X, idx_aug = select_augment(aug_values, aug_info, idx_train_valid)
            train_info, train_y = aug_info[idx_aug], aug_info[idx_aug,0]
            train_X = np.vstack([ nominal( train_X[i], 
                                           int(train_info[i,2]), 
                                           int(train_info[i,3]), 
                                           LOG 
                                        ) for i in range(len(train_X)) ])
            train_X = train_X[:,:,:TRUNC]

        if not PRIMOZ:
            # Densify
            N,P,T = train_X.shape
            
            train_avg = [
                    np.nanmean(train_X.reshape( (N, -1) )[train_info[:,2]==0],0),
                    np.nanmean(train_X.reshape( (N, -1) )[train_info[:,2]==1],0),
                    ]
            train_avg[0] = (np.isnan( train_avg[0] )+0)*org_avg[0] + np.nan_to_num( train_avg[0] )
            train_avg[1] = (np.isnan( train_avg[1] )+0)*org_avg[1] + np.nan_to_num( train_avg[1] )
            
            train_X = impute_and_features( train_X, train_info, train_avg, TRUNC )
            test_X  = impute_and_features( test_X , test_info , train_avg, TRUNC )
                
        if STANDARDIZE:
            scaler = StandardScaler()
            scaler.fit(train_X)
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)

        ##################
        # Evaluate
        #
        # RESULTS['v'] = np.zeros( (ITER, nCOEF, tCV, vCV, len(ALPHA_RANGE[im]), 3 + 9*im) )
        
        def run(n_mask):       
#        for n_mask in range(nCOEF)[::-1]:
            RETURN = []
            print(t2s(TPS), PREPROC, _iter+1,fold_test+1,n_mask+1,' '*10,end="\r",flush=True)
        
            MASK = np.arange(nCOEF) <= n_mask

            if ABLATION == 'bin':
                xx,xt = train_X[:,MASK],test_X[:,MASK]
                yy,yt = (train_y != 0)+0,(test_y != 0)+0
            elif ABLATION == 'ssi':
                xx,xt = train_X[train_y > 0][:,MASK],test_X[test_y > 0][:,MASK]
                yy,yt = train_y[train_y > 0]-1,test_y[test_y > 0]-1
            else:
                SSI = int(ABLATION[1])
                xx,xt = train_X[:,MASK],test_X[:,MASK]
                yy,yt = (train_y == SSI)+0,(test_y == SSI)+0
        
            MODEL = clone(_MODEL)
#            if n_mask <= 40:
#                MODEL.max_features=n_mask+1
            MODEL.fit(xx, yy)
            pred = MODEL.predict_proba(xt)
                
#            RESULTS['t'][n_mask, _iter, fold_test] = report_bin( yt, pred[:,1] )
#            RESULTS['l'][-1][-1].append( pred )
#            RESULTS['y'][-1][-1].append( yt )
            
            RETURN.append( report_bin( yt, pred[:,1] ) )
            RETURN.append( pred )
            RETURN.append( yt )

            N_SPLIT = max(1, int( (yt == 0).sum() / (yt != 0).sum() ) )
            split = np.zeros( len(yt) )
            split[ yt != 0 ] = -1
            temp = ( np.arange((yt==0).sum()) % N_SPLIT )
            np.random.shuffle(temp)
            split[ yt == 0 ] = temp
            rm = np.mean( [report_bin( 
                                yt[ (split==-1) + (split==s) ], 
                                pred[ (split==-1) + (split==s), 1] 
                            ) for s in range(N_SPLIT) ], 0 )
#            RESULTS['at'][n_mask, _iter, fold_test] = rm
#            RESULTS['al'][-1][-1].append( pred )
#            RESULTS['ay'][-1][-1].append( yt )
            
            RETURN.append( rm )
            RETURN.append( pred )
            RETURN.append( yt )
            
            return RETURN
            
        with Pool() as POOL:
            RETURN = POOL.map( run, range(nCOEF))
            
            for n_mask in range(nCOEF)[::-1]:
                RESULTS['t'][n_mask, _iter, fold_test] = RETURN[n_mask][0]
                RESULTS['l'][-1][-1].append( RETURN[n_mask][1] )
                RESULTS['y'][-1][-1].append( RETURN[n_mask][2] )
                
                RESULTS['at'][n_mask, _iter, fold_test] = RETURN[n_mask][3]
                RESULTS['al'][-1][-1].append( RETURN[n_mask][4] )
                RESULTS['ay'][-1][-1].append( RETURN[n_mask][5] )
#        np.save( sNAME, RESULTS )
        
        
RESULTS['l'] = np.array( RESULTS['l'] )
RESULTS['al'] = np.array( RESULTS['al'] )
RESULTS['y'] = np.array( RESULTS['y'] )
RESULTS['ay'] = np.array( RESULTS['ay'] )
np.save( sNAME, RESULTS )
print( t2s(TPS), '\n<<< Saved in:', sNAME, flush=True )   
        

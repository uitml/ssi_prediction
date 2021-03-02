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

import warnings
warnings.filterwarnings("ignore")

from utils import *

TPS = time.time()

if SAVE:
    if not os.path.isdir('save'):
        os.makedirs('save')
        
    FNAME = 'save/save_gboost_' + \
        PREPROC + \
        ( ('_t='+ str(TRUNC)) if not PRIMOZ else '' )+ \
        ('_log' if LOG else '') + \
        ('_ovs' if OVER_SAMPLE else '') + \
        ('_std' if STANDARDIZE else '') + \
        '.npy'
    print('*** I will save in: ',FNAME)

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
    day_values = np.vstack([ LOAD['x'], LOAD['tx'] ])[:,:,:STEP[-1]]
    day_info = np.vstack([ LOAD['i'], LOAD['ti'] ])
    del LOAD

    keep = ( np.isfinite(day_values).any(1).sum(1) >= MIN_WEEK_VALUES ) * ( day_info[:,2] != 1-GENDER )
    day_values = day_values[keep]
    day_info = day_info[keep]

    N,P,T = day_values.shape

    W = len(STEP) - 1
    # Fill the missing values of test-days using the average of the week.
    # Can be nan if a variable has not been tested that week
    # If a week had no test, add a dummy results to mark the week.
    for i in range(N):
        for s0,s1 in zip(STEP[:-1],STEP[1:]):
            if np.isnan(day_values[i,:,s0:s1]).all():
                day_values[i,:,s0] = -1
            else:
                m = np.nanmean(day_values[i,:,s0:s1],1)
                for s in np.isfinite( day_values[i,:,s0:s1] ).any(0).nonzero()[0]:
                    day_values[i,:,s0+s] = (np.isnan(day_values[i,:,s0+s])*m) + np.nan_to_num(day_values[i,:,s0+s])


RESULTS = {}
for im,m in enumerate(['bin','ssi','multi','combi']):
    RESULTS[m] = {}
    for p in ['','a']:
        RESULTS[m][p+'t'] = np.zeros( (ITER,tCV, 3 + 9*( im > 1 )) ) * np.nan
        RESULTS[m][p+'c'] = []
        RESULTS[m][p+'i'] = []
        RESULTS[m][p+'l'] = []
        RESULTS[m][p+'y'] = []


#for _iter in range(ITER):

def run(_iter):


    np.random.seed( _iter )
    random.seed( _iter )
    
    RETURN = {}
    for im,m in enumerate(['bin','ssi','multi','combi']):
        RETURN[m] = {}
        for p in ['','a']:
            RETURN[m][p+'t'] = np.zeros( (ITER,tCV, 3 + 9*( im > 1 )) )
            RETURN[m][p+'c'] = []
            RETURN[m][p+'i'] = []
            RETURN[m][p+'l'] = []
            RETURN[m][p+'y'] = []

    ################################
    # CROSS VALIDATION
    tFOLD = StratifiedKFold(n_splits=tCV, shuffle=True).split( range(org_info.shape[0]), org_info[:,0] )
    for fold_test, (idx_train_valid, idx_test) in enumerate(tFOLD):
    
        print(t2s(TPS), PREPROC, _iter+1,fold_test+1,' '*50,end="\r",flush=True)

        ################################
        # PREPARE TRAIN/TEST SETS

        # Split train/test
        test_X, test_info, test_y = org_values[idx_test], org_info[idx_test], org_info[idx_test, 0]
        if not AUGMENT:
            if OVER_SAMPLE:
                ovs = RandomOverSampler() if PRIMOZ else SMOTE()
                idx_train_valid = ovs.fit_resample(
                                                idx_train_valid[:,None], 
                                                org_info[idx_train_valid, 0]
                                            )[0].squeeze()
            train_X, train_info, train_y = org_values[idx_train_valid], org_info[idx_train_valid], org_info[idx_train_valid, 0]
            
        else:
            train_X, idx_aug = select_augment(day_values, day_info, idx_train_valid)
            train_info, train_y = day_info[idx_aug], day_info[idx_aug,0]
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
        # RESULTS[m]['v'] = np.zeros( (ITER,tCV, vCV, len(ALPHA_RANGE[im]), 3 + 9*im) )
        
        combi_pred = np.ones( (2, len(test_X), 3) )
        for im,MULTI in enumerate(['bin','ssi','multi'][:]):

            if MULTI == 'multi':
                x,y = train_X,train_y
                tx,ty = test_X,test_y
                report = report_multi
            elif MULTI == 'ssi':
                x,y = train_X[train_y > 0],train_y[train_y > 0]-1
                tx,ty = test_X[test_y > 0],test_y[test_y > 0]-1
                report = report_bin
            elif MULTI == 'bin':
                x,y = train_X,(train_y != 0)+0
                tx,ty = test_X,(test_y != 0)+0
                report = report_bin
                
            for ip,pfx in enumerate(['','a']):
            
                print(t2s(TPS), PREPROC, _iter+1,fold_test+1,MULTI,pfx,' '*10,end="\r",flush=True)
            
                N_SPLIT = (2 + 4 * (MULTI == 'multi') - (MULTI == 'ssi') -1) * (pfx == 'a') + 1

                MODEL = clone(_MODEL)
                MODEL.fit(x, y)
                pred = MODEL.predict_proba(tx)

                split = np.zeros( len(ty) )
                split[ ty != 0 ] = -1
                temp = ( np.arange((ty==0).sum()) % N_SPLIT )
                np.random.shuffle(temp)
                split[ ty == 0 ] = temp

                rm = np.mean( [report( 
                                    ty[ (split==-1) + (split==s) ], 
                                    pred[ (split==-1) + (split==s) ] 
                                ) for s in range(N_SPLIT) ], 0 )

                RETURN[MULTI][pfx + 't'][_iter,fold_test] = rm 
                RETURN[MULTI][pfx + 'c'].append( MODEL.feature_importances_ )
                RETURN[MULTI][pfx + 'l'].append( pred )            
                RETURN[MULTI][pfx + 'y'].append( ty )
                
                if MULTI == 'bin':
                    combi_pred[ip,:,0] *= pred[:,0]
                    combi_pred[ip,:,1] *= pred[:,1]
                    combi_pred[ip,:,2] *= pred[:,1]
                elif MULTI == 'ssi':
                    pred = MODEL.predict_proba(test_X)
                    combi_pred[ip,:,1] *= pred[:,0]
                    combi_pred[ip,:,2] *= pred[:,1]

                # DOES NOT WORK
#                IMPRT = permutation_importance( MODEL, tx, ty, n_repeats=10)
#                RESULTS[MULTI][pfx + 'c'].append( IMPRT.importances_mean )
#                RESULTS[MULTI][pfx + 'i'].append( IMPRT.importances_std )        
                
        MULTI = 'combi'
        x,y = train_X,train_y
        tx,ty = test_X,test_y
        report = report_multi
        for ip,pfx in enumerate(['','a']):
        
            print(t2s(TPS), PREPROC, _iter+1,fold_test+1,MULTI,pfx,' '*10,end="\r",flush=True)
                
            N_SPLIT = (2 + 4 * (MULTI == 'combi') - (MULTI == 'ssi') -1) * (pfx == 'a') + 1

            pred = combi_pred[ip]

            split = np.zeros( len(ty) )
            split[ ty != 0 ] = -1
            temp = ( np.arange((ty==0).sum()) % N_SPLIT )
            np.random.shuffle(temp)
            split[ ty == 0 ] = temp

            rm = np.mean( [report( 
                                ty[ (split==-1) + (split==s) ], 
                                pred[ (split==-1) + (split==s) ] 
                            ) for s in range(N_SPLIT) ], 0 )

            RETURN[MULTI][pfx + 't'][_iter,fold_test] = rm 
            RETURN[MULTI][pfx + 'l'].append( pred )
            RETURN[MULTI][pfx + 'y'].append( ty )
            
    return RETURN

      
#    print('Intermediate')
#    for im,MULTI in enumerate(['bin','multi']):
#        print( np.mean(RESULTS[MULTI]['a'][-fold_test:]) )
#        print( a2s((RESULTS[MULTI]['t'][_iter,:fold_test]).mean(0)[:3+3*im])  )
    ## ADD EVAL ON TEST SET using average weights of last cv np.mean(RESULTS['c'][ALGO][-CV:],0)
    # STORED in at
    
with Pool() as POOL:
    RETURN = POOL.map( run, range(ITER) )

    for _iter in range(ITER):
        for im,MULTI in enumerate(['bin','ssi','multi','combi']):
            for ip,pfx in enumerate(['','a']):
                RESULTS[MULTI][pfx + 't'][_iter] = RETURN[_iter][MULTI][pfx + 't'][_iter]
                RESULTS[MULTI][pfx + 'l'] += RETURN[_iter][MULTI][pfx + 'l']
                RESULTS[MULTI][pfx + 'y'] += RETURN[_iter][MULTI][pfx + 'y']
                RESULTS[MULTI][pfx + 'c'] += RETURN[_iter][MULTI][pfx + 'c']
#                RESULTS[MULTI][pfx + 'i'] += RETURN[_iter][MULTI][pfx + 'i']
                

print(' '*20 , end='\r', flush=True)

if SAVE:
    np.save( FNAME, RESULTS )
    print( '<<< Saved in:', FNAME, flush=True )

if VERBOSE > 0:    
    HEAD = '>>> '
    HEAD += (('' if AUGMENT else 'Non-') + 'Augmented Data') if not PRIMOZ else 'Primoz Features'
    HEAD += ('\n> Log Space' if LOG else '')
    HEAD += ('\n> Truncated at '+str(TRUNC)+' weeks' if not PRIMOZ else '')    
    HEAD += ('\n> Standardized' if STANDARDIZE else '')
    HEAD += ('\n> Random Over Sampling' if OVER_SAMPLE else '')
    HEAD += '\n> Gender = '+f_GENDER.title()
    HEAD += '.'
#    print(HEAD)
#    print('')

#    for t,ttl in [('v','Validation'),('t','Test')]:
    print('> Test', t2s(TPS))
    MEAN,SD = {},{}
    for k in RESULTS.keys(): 
        MEAN[k] = ( np.nanmean(RESULTS[k]['t'].reshape((ITER*tCV,-1)),0),
                    np.nanmean(RESULTS[k]['at'].reshape((ITER*tCV,-1)),0) )
        SD[k]   = ( np.nanstd (RESULTS[k]['t'].reshape((ITER*tCV,-1)),0), 
                    np.nanstd (RESULTS[k]['at'].reshape((ITER*tCV,-1)),0) )

#    print(' '*15 + 'Binary'+' '*9+'|'+' '*6+' Bin.  SSI '+' '*6+'|'+' '*6+'Multi-Class'+' '*6+'|'+' '*31+'One vs Rest',
#        )
    print(
        '\t AUPRC','\tAUROC','\t APS ',
        '|\t\tAUPRC\t     ','|\t\tAUROC\t     ','|\t\tAPS',
        )
    HLINE = '-'*20 + '|' +  \
        '-'*4 + '0' + '-'*6 + '1' + '-'*7 + '2' + '-'*3 + '|' + \
        '-'*4 + '0' + '-'*6 + '1' + '-'*7 + '2' + '-'*3 + '|' + \
        '-'*4 + '0' + '-'*6 + '1' + '-'*7 + '2' + '-'*3 + '|' 

    ok = ''
    for im,m in enumerate(['bin','ssi','multi','combi'][:]):
        # im += 2
        print( m.title() + '-'*(10-len(m)) + HLINE )
        
        for i,b in enumerate(['imbal','avg']):
            if im < 2:
                print(b,
                    '\t', 
                    a2s(MEAN[m][i][:3])
                )
            else:
                print(b,
                    '\t', 
                    a2s(MEAN[m][i][:3]),
                    '|\t',
                    a2s(MEAN[m][i][-9:-6]),
                    '|\t',
                    a2s(MEAN[m][i][-6:-3]),
                    '|\t',
                    a2s(MEAN[m][i][-3:]),
                )
    print('-'*102 + '|')

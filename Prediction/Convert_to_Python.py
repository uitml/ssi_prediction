import copy
import numpy as np
import pandas as pd
from datetime import datetime

def get_onset(d):
    try:
        ds = datetime.strptime(d[0],"%Y-%m-%d %H:%M:%S")
        di = datetime.strptime(d[1],"%Y-%m-%d %H:%M:%S")
        return (di-ds).days
    except TypeError:
        return np.nan

FOLDER = 'data/'

print('>>> Prepare Primoz data.')

def proc_primoz(FOLD):
    print('>> Process '+FOLD+'.')
    
    LABELS = pd.read_csv( FOLDER + FOLD+'_Labels.csv' )
    Y = LABELS.Infection.to_numpy().astype(int)
    I = LABELS.loc[:,['Infection','PID','Sex','Age']].to_numpy()
    I[:,2] = (I[:,2] == 'M') + 0
    I = I.astype(int)
    if 't.Infection' in LABELS.columns:
        OS = np.array([ get_onset(d) for d in LABELS.loc[:,['t.IndexSurgery','t.Infection']].to_numpy().tolist() ])
    else:
        OS = np.nan * np.ones(len(I))
    X  = pd.read_table( FOLDER + FOLD + '_Data_Primoz.tsv' ).to_numpy()[:,1:]
    return X, Y, I, OS

PRZ,Y,I,OS = proc_primoz('Train')
tPRZ,tY,tI,tOS = proc_primoz('Eval')
np.savez( FOLDER + 'Data_Primoz.npz', x=PRZ, y=Y, i=I, os=OS , tx=tPRZ, ty=tY, ti=tI, tos=tOS)
del PRZ,tPRZ


print('>>> Prepare Daily data.')

def proc_daily(FOLD):
    print('>> Process '+FOLD+'.')

    RAW = pd.read_csv( FOLDER + FOLD+'_Raw.csv' )
    PID = np.unique(RAW.PID)
    PID.sort()
    PID = list(PID)

    TEST = np.unique(RAW.TestType)
    TEST.sort()
    TEST = list(TEST)
    N,T = len(PID),len(TEST)

    X = np.nan * np.zeros( (N,T,61) )
    for _,_i,_t,c,v in RAW.to_numpy():
        i = PID.index(_i)
        t = TEST.index(_t)
        X[i,t,-c] = v
        
    LABELS = pd.read_csv( FOLDER + FOLD+'_Labels.csv' )

    Y = LABELS.Infection.to_numpy().astype(int)
    I = LABELS.loc[:,['Infection','PID','Sex','Age']].to_numpy()
    I[:,2] = (I[:,2] == 'M') + 0
    FD = X.shape[2]-1-np.isfinite(X).any(1)[:,::-1].argmax(1)
    I = np.hstack([I,FD[:,None]]).astype(int)
    if 't.Infection' in LABELS.columns:
        OS = np.array([ get_onset(d) for d in LABELS.loc[:,['t.IndexSurgery','t.Infection']].to_numpy().tolist() ])
    else:
        OS = np.nan * np.ones(len(I))
    return X, Y, I, OS

DAILY,Y,I,OS = proc_daily('Train')
tDAILY,tY,tI,tOS = proc_daily('Eval')
np.savez( FOLDER + 'Data_Daily.npz', x=DAILY, y=Y, i=I, os=OS, tx=tDAILY, ty=tY, ti=tI, tos=tOS )


print('>>> Prepare Weekly data.')
# Partition of the 1+2+7 weeks before OP
STEP = [0,1,4] + list(range(8,61,7))
# STEP = [1,4] + list(range(8,61,7))
# Remove day=0 (Surgery)
#STEP = [1,2,4] + list(range(8,61,7))

def proc_weekly(FOLD,DAILY):
    print('>> Process '+FOLD+'.')
    N,P,T = DAILY.shape
    W = len(STEP)-1
    X = np.nan * np.zeros((N,P,W))
    for s, (s0,s1) in enumerate(zip(STEP[:-1],STEP[1:])):
        X[:,:,s] = np.nanmean(DAILY[:, :, s0:s1], 2)
    return X
    
WEEKLY = proc_weekly('Train', DAILY)
tWEEKLY = proc_weekly('Eval', tDAILY)
np.savez( FOLDER + 'Data_Weekly.npz', x=WEEKLY, y=Y, i=I, os=OS, tx=tWEEKLY, ty=tY, ti=tI, tos=tOS )
del WEEKLY


print('<<< Done.')

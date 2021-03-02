import os,sys,copy,string,random,time
import numpy as np
import pandas as pd
import argparse

from blood_samples import nominal_values

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler 

from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.linear_model import LassoLarsIC, SGDClassifier
from sklearn.metrics import precision_score as precis
from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import average_precision_score as aps
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score  as recall
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

# short = True
# folder = "/data/" if os.path.isdir("/data/") else "./data/"
# datafile = folder + "dataset_short" if short else folder + "dataset" 
datafile = "dataset_short"

TEST = ['Hemoglobin', 'Leukocytter', 'Natrium', 'CRP', 'Kalium', 'Albumin', 'Kreatinin', 'Trombocytter', 'ALAT', 'Bilirubin total', 'ASAT', 'ALP', 'Amylase', 'Glukose']
EXTRA = ['SSI','Gender','Age','First test']
COLOR = ['gray','steelblue','tomato','seagreen']
LGD = ['SSI=0','SSI=1','SSI=2','All']

STEP = [0,1,4] + list(range(8,61,7))
# STEP = [1,4] + list(range(8,61,7))
# STEP = [1,2,4] + list(range(8,61,7))

ALPHA_RANGE = np.linspace(-4, 4, 9)
# ALPHA_RANGE = np.linspace(-2,0, 9)

###################################################################

parser = argparse.ArgumentParser()
#parser.add_argument("--algo", 
#                        help="Which algo?",
#                        choices=['sgdlog', 'gboost'],
#                        default='sgdlog',  
#                        type=str,
#                    )
parser.add_argument("preproc", 
                        help="Type of Pre-processing.",
                        choices=['aug', 'org', 'prz'], 
                        default='org', 
                        type=str,
                    )
parser.add_argument("-l","--log", 
                        help="Convert values to log space.",
                        action="store_true",
                    )
parser.add_argument("-ovs","--over_sample", 
                        help="Random over Sampling? Default for Primoz. Not compatible with augmentation.",
                        action="store_true",
                    )
parser.add_argument("-std","--standardize", 
                        help="Standardize the data? Default for Primoz.",
                        action="store_true",
                    )
parser.add_argument("-i","--iter", 
                        help="Number of iterations.",
                        default=10, 
                        type=int, 
                    )
parser.add_argument("-tcv","--cv_test", 
                        help="Number of folds of the test Cross-validation.",
                        default=10, 
                        type=int, 
                    )
parser.add_argument("-vcv","--cv_valid", 
                        help="Number of folds of the vatidation  Cross-validation.",
                        default=5, 
                        type=int, 
                    )
parser.add_argument("-as", "--aug_sample", 
                        help="Number of randomly drawn augmented samples.",
                        default=5, 
                        type=int, 
                    )
parser.add_argument("-g", "--gender", 
                        help="Gender.",
                        choices=['all','a','A', 'female','f','F', 'male','m','M'], 
                        default='all', 
                        type=str, 
                    )
parser.add_argument("-t","--trunc", 
                        help="Truncation of the data. If not given, it is set to t=4/7/2 for gender=All/Female/Male, respectively.",
                        choices=list(range(1,11)), 
                        default=-1, 
                        type=int, 
                    )
parser.add_argument("-ab","--ablation",
                        help="ONLY FOR ABLATION: for binary, ssi or multi_class",
                        choices=['bin','ssi','m0','m1','m2'], 
                        default='bin', 
                        type=str, 
                    )
parser.add_argument("-v","--verbose", 
                        help="Level of verbosity.",
                        default=1, 
                        type=int, 
                    )
parser.add_argument("-r","--random_seed", 
                        help="Random Seed.",
                        default=0, 
                        type=int, 
                    )
parser.add_argument("-s","--save", 
                        help="Save predictions and coefficients.",
                        default=1,
                        type=int,
                    )
                    
    


###################################################################
if ".py" in parser.prog:
    args = parser.parse_args()

    if args.preproc in ['a','aug','1']:
        PREPROC = 'aug'
        AUGMENT = True
        PRIMOZ  = False
        ALPHA   = (-1.65,-1.389)
    elif args.preproc in ['o','org','0']:
        PREPROC = 'org'
        AUGMENT = False
        PRIMOZ  = False
        ALPHA   = (-1.765,-1.647)
    elif args.preproc in ['prz','p','2']:
        PREPROC = 'prz'
        AUGMENT = False
        PRIMOZ  = True
        ALPHA   = (-1.677,-1.97)
    else:
        print('!!! arg should be: org|aug|prz !!!')
        sys.exit()

    AUG_SAMPLE = args.aug_sample #8 # GRID SEARCH # 10 # 
    ITER = args.iter # 100 # 
    tCV = args.cv_test # 10 # 
    vCV = args.cv_valid # 5 # 
    MIN_WEEK_VALUES = 1

    if args.over_sample and AUGMENT:
        print('!!! You cannot both OVER SAMPLE and AUGMENT the data.')

    LOG = args.log if not PRIMOZ else False # ~PRIMOZ # 
    OVER_SAMPLE = ( (args.over_sample and  not AUGMENT) or PRIMOZ) > 0 # ~AUGMENT # 
    STANDARDIZE = ( args.standardize + PRIMOZ > 0) # True # 

    GENDER = ['F','M','A'].index(  args.gender[0].upper() ) # 2 # 
    h_GENDER = ['F','M','A'][GENDER]
    f_GENDER = ['female','male','all'][GENDER]

    # ALGO = args.algo

    VERBOSE = args.verbose
    SAVE = args.save == 1

    if args.trunc == -1:
        if GENDER == 0:
            TRUNC = 4
        elif GENDER == 1:
            TRUNC = 6
        else:
            TRUNC = 6
    else:
        TRUNC = args.trunc
        
    ABLATION = args.ablation # [int(x) for x in args.ablation.split(',')]
     
    if args.random_seed > -1:
        RANDOM_SEED = args.random_seed
        np.random.seed( RANDOM_SEED )
        random.seed( RANDOM_SEED )
else:

    PREPROC = 'org'
    AUGMENT = False 
    PRIMOZ  = False  
    AUG_SAMPLE = 20  
    ITER = 2
    tCV = 2
    vCV = 2
    LOG = True  
    MIN_WEEK_VALUES = 1
    OVER_SAMPLE = True
    STANDARDIZE = True
    VERBOSE = 2  
    RANDOM_SEED = 0 
    np.random.seed( RANDOM_SEED ) 
    random.seed( RANDOM_SEED )
    TRUNC = 6
    GENDER = -1
    h_GENDER = ['F','M','A'][GENDER]
    f_GENDER = ['female','male','all'][GENDER]
    SAVE = False
    ALGO = 'gboost'
    ABLATION = 'bin'


# h_ALGO = [ ALGO, 'M '+ALGO]
         
#if ALGO == 'sgdlog':
#    _MODEL = SGDClassifier(
#                loss='log', 
#                penalty='l1', 
#                n_jobs=-1, 
#                learning_rate='adaptive',
#                eta0 = 1e-5, 
#            )
#else:
#    _MODEL = GradientBoostingClassifier(
#                loss='deviance',
#                learning_rate=.05, 
#                n_estimators=30, 
#                max_depth=5, 
#                subsample=1.,
#                criterion='mse',
#                max_features='sqrt',
#                min_samples_split=50,
#                min_samples_leaf=5,
#            )

STEP = STEP[:TRUNC+1]




################################
# UTILS

def a2s(array): 
    return str( np.round(array,3).tolist() )[1:-1].replace(', ','\t') 
    
def t2s(tps):
    dt = time.time() - tps
    h = int(dt/60/60)
    m = int(dt - h*60*60)/60
    s = int(dt - h*60*60 - m*60 )
    return '{:.0f}:{:.0f}:{:.0f}'.format(h,m,s)
    
    
def auprc(true, pred):
    precision, recall, thresholds = precision_recall_curve(true, pred)
    return auc(recall, precision)

def report_bin(true, pred):
    if len(pred.shape) == 2:
        pred = pred[:,1]
    bpred = np.round(pred)
    bpred[ bpred>1 ] = 1
    bpred[ bpred<0 ] = 0
    return [ 
        auprc(true, pred) ,
        auroc(true, pred) ,
        aps(true, pred) ,
#        precis(true, bpred, pos_label=1),
#        precis(true, bpred, pos_label=0),
#        recall(true, bpred, pos_label=1),
#        recall(true, bpred, pos_label=0),
#        f1_score(true, bpred, pos_label=1),
#        f1_score(true, bpred, pos_label=0),
    ]


def report_multi(true, pred, avg='micro'):
    if np.isnan(pred).any():
#        return (np.nan,) * len( report_multi([0,0],[0,0]) )
        return (np.nan,) * 11
        
    true, pred = np.array(true), np.array(pred)
    mx = int(max(true)+2)
    b_true = label_binarize(true, range(mx) )[:,:-1]
    if len(pred.shape) == 1:
        pred[ pred>mx-1 ] = mx-1
        pred[ pred<0 ] = 0
        pred = np.round(pred)
        pred = label_binarize(pred, range(mx) )[:,:-1]
        b_pred = pred
    else:
        b_pred = label_binarize(np.argmax(pred,1), range(mx) )[:,:-1]    
    
    R = [ 
        auprc(true!=0, 1-pred[:,0]),
        auroc(b_true, pred, multi_class='ovr',average=avg) ,
        aps(b_true, pred, average=avg) ,
#        precis(b_true, b_pred, average=avg),
#        recall(b_true, b_pred, average=avg),
#        f1_score(b_true, b_pred, average=avg),
    ] + \
    [auprc( true==ssi, pred[:,ssi]) for ssi in range(3)] + \
    auroc(b_true, pred, multi_class='ovr',average=None).tolist() + \
    aps(b_true, pred, average=None).tolist()
    
    return R



################################
# PREPROC
    
def draw_augment_old(x,n):
    P,W = len(x), len(STEP)-1
    nna = [ np.isfinite(x[:,s0:s1]).any(0).nonzero()[0]+s0 for (s0,s1) in zip(STEP[:-1],STEP[1:]) ]
    p = np.array([ len(nn) for nn in nna])
    t = []
    for _ in range(n):
        r = np.round( np.random.random( W ) * (p-1) ).astype(int)
        t.append( np.zeros((P,W)) )
        for s in range(W):
            t[-1][:,s] = x[:, nna[s][ r[s] ] ]
    return t


    
def draw_augment(x,n):
    P,W = len(x), len(STEP)-1
    nna = [ np.isfinite(x[:,s0:s1]).any(0).nonzero()[0]+s0 for (s0,s1) in zip(STEP[:-1],STEP[1:]) ]
    t = []
    for _ in range(n):
        t.append( np.zeros((P,W)) )
        for s in range(W):
            r = list(np.random.choice( len(nna[s]), np.random.randint(1, len(nna[s])+1 ) ))
            t[-1][:,s] = np.nanmean( x[:, nna[s][ r ] ], 1)
    return t

def select_augment(day_values, info, idx_train):
    global AUG_SAMPLE
    data,idx = [],[]
    for i in idx_train:
        spl   = 1 if info[i, 0] == 0 else AUG_SAMPLE
        data += draw_augment(day_values[i], spl)
        idx  += [i]*spl
    data = np.array(data)
    return data, idx

def nominal(x, gender, age, log=False):
    P,T = x.shape
    
    normal_values = nominal_values(TEST, ['F','M'][gender], age)
    normal_values[ normal_values == 0] = 1
    if log:
        normal_values = np.log(normal_values)
        x = np.log(x)
    
    m = normal_values.mean(1)
    s = np.diff( normal_values, 1) / 2
    m = np.array([m]*T).T
    s = np.array([s]*T).T
    
    x = (x - m) / s
        
    return x

def set_nominal(x, m, s, log=False):
    if log:
        x = np.log(x)    
    x = (x - m.T) / s.T
    return [x]

def impute_and_features(values, info, avg=None, trunc=TRUNC):
    
    N,P,T = values.shape
    nb = np.sum(np.isfinite(values),1)
    SSI = int(info[:,0].max())+1
    
    values = values.reshape( (N, -1) )
    
    if avg is None:
        avg = (
                np.nanmean(values[info[:,1]==0],0),
                np.nanmean(values[info[:,1]==1],0)
            )
                    
    for gender in range(2):
        for ssi in range(SSI):
            selec  = (info[:,0] == ssi) * (info[:,2]==gender)
            if selec.any():
                nn = np.isnan(values[selec]).all(0).nonzero()[0]
                for j in nn:
                    values[selec,j] = avg[gender][j]
                values[selec] = KNNImputer(n_neighbors=5).fit_transform( values[selec] )
    values = values.reshape( (N, P, T) )
    
#    values = values[:,:,:TRUNC]
    
    
    temp = np.hstack([ values.reshape((N,-1)) , -np.diff( values ,1).reshape((N,-1)), info[:,2:] ])
    if LOG:
        temp[:,-1] = np.log10(1 + info[:,-1].astype('float'))
        
    return temp

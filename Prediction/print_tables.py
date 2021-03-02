import argparse,random
import numpy as np
from scipy.stats import rankdata 

# from utils import *
import argparse

def a2s(array): 
    return str( np.round(array,3).tolist() )[1:-1].replace(', ','\t')

STEP = [0,1,4] + list(range(8,61,7))

###################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-l","--latex", 
                        help="Print for LateX",
                        action="store_true",
                    )

args = parser.parse_args()

LATEX = args.latex
SEP = ' & ' if LATEX else '\t'
    

###################################################################

FNAME = [
    'save/save_gboost_prz_ovs_std.npy',
    'save/save_sgdlog_opt_l2_prz_ovs_std.npy',
    'save/save_gboost_org_t=6_log_ovs_std.npy',
    'save/save_sgdlog_opt_l2_org_t=6_log_ovs_std.npy',
#    'save/save_gboost_aug_t=5_log_std.npy',
#    'save/save_sgdlog_aug_t=5_log_std.npy'
    ]
    
ALGO = ['\\GBoost', '\\SGD  ',
        '\\GBoost', '\\SGD  ',
        '\\GBoost', '\\SGD  ']
        
TRUNC = 6

#    for t,ttl in [('v','Validation'),('t','Test')]:

MEAN,SD = {},{}
for f,FEAT in enumerate(['prz+g','prz+s','org+g','org+s']): #,'aug+g','aug+s']):
    MEAN[FEAT] = {}
    SD[FEAT] = {}
    RESULTS = np.load(FNAME[f],allow_pickle=True).item()
    for im,MULTI in enumerate(['bin','ssi','multi','combi']): 
        MEAN[FEAT][MULTI] = np.nanmean(RESULTS[MULTI]['at'].reshape((-1,3+9*(im>1))),0).reshape((-1,3))
        SD[FEAT][MULTI]   = np.nanstd (RESULTS[MULTI]['at'].reshape((-1,3+9*(im>1))),0).reshape((-1,3))
        if im > 1:
            MEAN[FEAT][MULTI][1:] = MEAN[FEAT][MULTI][1:].T
            SD[FEAT][MULTI][1:]   = SD[FEAT][MULTI][1:].T
        MEAN[FEAT][MULTI] = MEAN[FEAT][MULTI][:,:2]
        SD[FEAT][MULTI]   = SD[FEAT][MULTI][:,:2]
    

# Print Table 1
if LATEX:
    print('&          & AUPRC                 & AUROC                 & AUPRC                 & AUROC                 & AUPRC                 & AUROC \\\\ \\hline')
else:
    print(' '*45 + 'SSI' + ' '*36 + 'DEPTH' + ' '*32 + 'MULTI-CLASS')
    print(' '*36 + 'AUPRC' + ' '*11 + 'AUROC' + \
    ' '*20 + 'AUPRC' + ' '*10 + 'AUROC' + \
    ' '*20 + 'AUPRC' + ' '*10 + 'AUROC' )
       
for f,FEAT in enumerate(['prz+g','prz+s','org+g','org+s']): #,'aug+g','aug+s']):
    LINE = ('Kocbek  ' if FEAT[0] == 'p' else ('Ours+aug' if FEAT[0] == 'a' else 'Ours+ovs'))
    LINE += SEP + ALGO[f]
    for im,MULTI in enumerate(['bin','ssi','combi']): 
        LINE += '\t'
        for s in range(2):
            LINE += SEP
            if LATEX:
                LINE += '$'
            LINE += ( '\\mathbf{' if LATEX and f == 2 else '' )
            LINE += '{:.3f}'.format(MEAN[FEAT][MULTI][0,s])
            LINE += ( '}$ $\\mathbf{' if LATEX and f == 2 else '$ $' )
            LINE += ( '\\pm ' if LATEX else '(' )
            LINE += '{:.3f}'.format(SD[FEAT][MULTI][0,s])
            LINE += ( '}' if LATEX and f == 2 else '' )
            LINE += ( '$' if LATEX else ')' )
    LINE += ' \\\\'
    print(LINE)


#print('\n> Multi')

#MULTI = 'multi'
#if LATEX:
#    print("""& &  
#AUPRC               & AUROC 
#AUPRC               & AUROC 
#AUPRC               & AUROC \\\\ \\hline""")
#else:
#    print(' '*12 + 'AUPRC'+ ' '*10 + 'AUROC')
## Print Table 2
#for f,FEAT in enumerate(['prz+g','prz+s','org+g','org+s']): #,'aug+g','aug+s']):
#    LINE = ('Kocbek  ' if FEAT[0] == 'p' else ('Ours+aug' if FEAT[0] == 'a' else 'Ours+org'))
#    LINE += SEP + ALGO[f] + '\n'
#    for SSI in range(1,4):
#        LINE += ('' if LATEX else 'SSI='+str(SSI-1))
#        for s in range(2):
#            LINE += SEP
#            if LATEX:
#                LINE += '$'
#            LINE += '{:.3f}'.format(MEAN[FEAT][MULTI][SSI, s])
#            LINE += ( '$ $\\pm ' if LATEX else '(' )
#            LINE += '{:.3f}'.format(SD[FEAT][MULTI][SSI, s])
#            LINE += ( '$' if LATEX else ')' )
#        LINE += '\n'
#    LINE += ' \\\\'
#    if f %2 == 1:
#        LINE += ' \\hline'
#    print(LINE)


print('\n> Combi')

MULTI = 'combi'
if LATEX:
    print("""& &  
AUPRC               & AUROC 
AUPRC               & AUROC 
AUPRC               & AUROC \\\\ \\hline""")
else:
    print(' '*12 + 'AUPRC'+ ' '*10 + 'AUROC')
# Print Table 2
for f,FEAT in enumerate(['prz+g','prz+s','org+g','org+s']): #,'aug+g','aug+s']):
    LINE = ('Kocbek  ' if FEAT[0] == 'p' else ('Ours+aug' if FEAT[0] == 'a' else 'Ours+org'))
    LINE += SEP + ALGO[f] + '\n'
    for SSI in range(1,4):
        LINE += ('' if LATEX else 'SSI='+str(SSI-1))
        for s in range(2):
            LINE += SEP
            if LATEX:
                LINE += '$'
            LINE += ( '\\mathbf{' if LATEX and f == 2 else '' )
            LINE += '{:.3f}'.format(MEAN[FEAT][MULTI][SSI, s])
            LINE += ( '}$ $\\mathbf{' if LATEX and f == 2 else ('$ $' if LATEX else ''))
            LINE += ( '\\pm ' if LATEX else '(' )
            LINE += '{:.3f}'.format(SD[FEAT][MULTI][SSI, s])
            LINE += ( '}' if LATEX and f == 2 else '' )
            LINE += ( '$' if LATEX else ')' )
        LINE += '\n'
    LINE += ' \\\\'
    if f %2 == 1:
        LINE += ' \\hline'
    print(LINE)
    
print('\n')

# Print Table 3

aSTEP = np.array(STEP)
TEST = ['Hemoglobine', 'Leukocyttes', 'Sodium', 'CRP', 'Potassium', 'Albumine', 'Creatinine', 'Trombocyttes', 'ALAT', 'Bilirubine tot', 'ASAT', 'ALP', 'Amylase', 'Glucose']
LBL = np.array(
        [ t +' '*(20-len(t))+ SEP + str(aSTEP[i])+'-'+str(aSTEP[i+1]-1) for i in range(TRUNC) for t in TEST ]+ \
        [ '$\\Delta$' + t+' '*(10-len(t)) + SEP + str(aSTEP[i])+'-'+str(aSTEP[i+1]-1) for i in range(TRUNC-1) for t in TEST ]+ \
        ['Gender' + SEP + '-','Age' + SEP + '-','1st Day'+ ' '*13 + SEP + '-']
    )
    

RESULTS = np.load(FNAME[-2],allow_pickle=True).item()
for MULTI in ['bin','ssi']:
    rCOEF,rSIGN = [],[]
    for wgt in RESULTS[MULTI]['c']:
        rCOEF.append( rankdata( -np.abs(wgt) ) )
    rCOEF = np.array( rCOEF )

    sgn = np.sign(np.array(RESULTS[MULTI]['c']))
    rSIGN = np.vstack([(sgn==-1).mean(0), (sgn==0).mean(0), (sgn==1).mean(0)]).T

    print("\n"+MULTI.upper())

    if LATEX:
        print('Feature','Days','Avg.Rank','Sign',sep=SEP)
    else:
        print('Feature'+' '*10,'Days'+' '*1,'Avg.Rank (std)','Sign',sep=SEP)
        print('-'*50)
    for i in  np.mean(rCOEF,0).argsort()[:10]: 
        nm = LBL[i].replace('0-0','0')
        if LATEX:
            LINE = nm
            LINE += SEP
            LINE += "$ " + str(np.round(rCOEF[:,i].mean(0),2)) 
            LINE += ' \\pm '+str(np.round(rCOEF[:,i].std(0),2))+' $'
            LINE += SEP
            LINE += ['-','0','+'][ rSIGN[i].argmax() ]
            LINE += SEP
            print(LINE)
        else:
            print(
                    str(i)+nm,
                    str(np.round(rCOEF[:,i].mean(0),2)),
                    '('+str(np.round(rCOEF[:,i].std(0),2))+')', 
                    ['-','0','+'][ rSIGN[i].argmax() ],
                    sep=SEP
                    )

#Bilirubin tot   & 1-3 &     $ 20.79 \pm 24.6 $ & - &    $\Delta$Natrium     & 0 &       $ 37.51 \pm 30.3 $ & - \\ 

for SSI in range(3):
    print('\nSSI=',SSI)

    if LATEX:
        print('Feature','Days','Avg.Rank','Sign'+(' \\\\ \\hline' if SSI != 1 else ''),sep=SEP)
    else:
        print('Feature'+' '*10,'Days'+' '*1,'Avg.Rank (std)','Sign',sep=SEP)
        print('-'*50)

    rCOEF,rSIGN = [],[]
    for wgt in RESULTS['multi']['c']:
        rCOEF.append( rankdata( -np.abs(wgt[SSI]) ) )
    rCOEF = np.array( rCOEF )
    sgn = np.sign(np.array(RESULTS['multi']['c'])[:,SSI])
    rSIGN = np.vstack([(sgn==-1).mean(0), (sgn==0).mean(0), (sgn==1).mean(0)]).T
    
    for i in  np.mean(rCOEF,0).argsort()[:10]: 
        nm = LBL[i].replace('0-0','0')
        if LATEX:
            LINE = nm
            LINE += SEP
            LINE += "$ " + str(np.round(rCOEF[:,i].mean(0),2)) 
            LINE += ' \\pm '+str(np.round(rCOEF[:,i].std(0),2))+' $'
            LINE += SEP
            LINE += ['-','0','+'][ rSIGN[i].argmax() ]
            LINE += ( SEP if SSI == 1 else ' \\\\ ' )
            print(LINE)
        else:
            print(
                    nm,
                    str(np.round(rCOEF[:,i].mean(0),2)),
                    '('+str(np.round(rCOEF[:,i].std(0),2))+')', 
                    ['-','0','+'][ rSIGN[i].argmax() ],
                    ' \\\\ ' if SSI%2 == 0 else '',
                    sep=SEP,
                    )
    

###################################################################

FNAME = [
    'save/save_gboost_org_t=6_std.npy',
    'save/save_gboost_org_t=6_log_std.npy',
    'save/save_gboost_org_t=6_ovs_std.npy',
    'save/save_gboost_org_t=6_log_ovs_std.npy',
#    'save/save_gboost_aug_t=5_log_std.npy',
#    'save/save_sgdlog_aug_t=5_log_std.npy'
    ]
    
ALGO = ['\\GBoost', '\\GBoost',
        '\\GBoost', '\\SGD  ',
        '\\GBoost', '\\SGD  ']
        
TRUNC = 6

#    for t,ttl in [('v','Validation'),('t','Test')]:

MEAN,SD = {},{}
for f,FEAT in enumerate(['org','org+log','org+ovs','org+log+ovs']): #,'aug+g','aug+s']):
    MEAN[FEAT] = {}
    SD[FEAT] = {}
    RESULTS = np.load(FNAME[f],allow_pickle=True).item()
    for im,MULTI in enumerate(['bin','ssi','multi','combi']): 
        MEAN[FEAT][MULTI] = np.nanmean(RESULTS[MULTI]['at'].reshape((-1,3+9*(im>1))),0).reshape((-1,3))
        SD[FEAT][MULTI]   = np.nanstd (RESULTS[MULTI]['at'].reshape((-1,3+9*(im>1))),0).reshape((-1,3))
        if im > 1:
            MEAN[FEAT][MULTI][1:] = MEAN[FEAT][MULTI][1:].T
            SD[FEAT][MULTI][1:]   = SD[FEAT][MULTI][1:].T
        MEAN[FEAT][MULTI] = MEAN[FEAT][MULTI][:,:2]
        SD[FEAT][MULTI]   = SD[FEAT][MULTI][:,:2]
    

print('\n')

# Print Table 1
if LATEX:
    print('&          & AUPRC                 & AUROC                 & AUPRC                 & AUROC                 & AUPRC                 & AUROC \\\\ \\hline')
else:
    print(' '*45 + 'SSI' + ' '*36 + 'DEPTH' + ' '*32 + 'MULTI-CLASS')
    print(' '*36 + 'AUPRC' + ' '*11 + 'AUROC' + \
    ' '*20 + 'AUPRC' + ' '*10 + 'AUROC' + \
    ' '*20 + 'AUPRC' + ' '*10 + 'AUROC' )
       
for f,FEAT in enumerate(['org','org+log','org+ovs','org+log+ovs']): #,'aug+g','aug+s']):
    LINE = FEAT.title()
    LINE += SEP if (f != 3 and not LATEX) else ''
    LINE += SEP + ALGO[f]
    for im,MULTI in enumerate(['bin','ssi','combi']): 
        LINE += '\t'
        for s in range(2):
            LINE += SEP
            if LATEX:
                LINE += '$'
            LINE += ( '\\mathbf{' if LATEX and f == 3 else '' )
            LINE += '{:.3f}'.format(MEAN[FEAT][MULTI][0,s])
            LINE += ( '}$ $\\mathbf{' if LATEX and f == 3 else '$ $' )
            LINE += ( '\\pm ' if LATEX else '(' )
            LINE += '{:.3f}'.format(SD[FEAT][MULTI][0,s])
            LINE += ( '}' if LATEX and f == 3 else '' )
            LINE += ( '$' if LATEX else ')' )
    LINE += ' \\\\'
    print(LINE)

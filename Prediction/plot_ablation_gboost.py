import argparse,random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def a2s(array): 
    return str( np.round(array,3).tolist() )[1:-1].replace(', ','\t') 


sns.set_style("whitegrid")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['legend.fontsize'] = 20

FS1 = 20
FS2 = 20
FS3 = 18
MS1 = 32
MS2 = 128
MS3 = 64
LWD1= 2
LWD2= 3

COLOR = ['b','g','r','orange','cyan','cyan','olive','y']
MARK = ['.','+','s','*','v','purple','olive','y']

COLOR = sns.color_palette('Set1')
COLOR = ['steelblue','orange','seagreen','tomato','seagreen']
#COLOR = ['seagreen','orange','tomato','orchid','steelblue']
#COLOR = ['seagreen', 'orange', 'tomato', 'orchid', 'steelblue', 'gold', 'firebrick']
nC = len(COLOR)

###################################################################

TITLE = ['Binary','No Infection','Shallow Infection','Deep Infection']
TITLE = ['Binary','One-vs-Rest']

LBL = ['SSI','Depth','None','Shallow','Deep']

plt.figure(1,(8,5),300)

for i,SSI in enumerate(['bin','ssi']):

#    plt.subplot(1,2,i+1)
    RESULTS = np.load('save/ablation_save_gboost_org_t=6_log_ovs_std_ssi='+SSI+'.npy', allow_pickle=True).item()
    X = RESULTS['at'].reshape((157,-1,3)).mean(1)
    print(X.shape)
    plt.plot([0],c='w',label=LBL[i])
    plt.plot([0],c='w',label=" ")
    plt.plot(X[:,1][::-1],'-',c=COLOR[0+2*i],label='AUROC', linewidth=2.5) 
    plt.plot(X[:,0][::-1],'-',c=COLOR[1+2*i],label='AUPRC', linewidth=2.5)
#    if i == 0:
#        plt.plot([0],c='w',label=LBL[i])
    
#plt.title(TITLE[0], fontsize=FS2)
        
#tk = np.array([0]+ list(range(27,150,30))+[156])
tk = np.array([0]+ list(range(27,120,30)) + list(range(127,156,10)) + [156])
#plt.xticks( tk[:-1], (156-tk+1)[:-1],fontsize=FS3)
#plt.xticks( tk[:], (tk.max()-tk+1)[:5],fontsize=FS3)
plt.xticks( tk[:],  (tk.max()-tk+1) ,fontsize=FS3)
plt.xlim(0, tk.max())
plt.xlabel('Number of features left', fontsize=FS3)
plt.ylabel('AUPRC / AUROC', fontsize=FS3)

yl = np.linspace(.6,1,5)
plt.ylim( yl[0], yl[1])
plt.yticks( yl,fontsize=FS3) 

plt.subplots_adjust(.1,.33,.95,.95,.2,.0); 
    
plt.legend(loc=8,ncol=4,prop={ 'size':FS3},bbox_to_anchor=(.5,-.525),columnspacing=.5)

plt.savefig('ablation_gb2')
plt.close()

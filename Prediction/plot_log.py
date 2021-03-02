import argparse,random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from scipy.optimize import minimize

from blood_samples import nominal_values

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
COLOR = ['steelblue','orange','seagreen','tomato','orchid']
COLOR = ['seagreen','orange','tomato','orchid','steelblue']
COLOR = ['seagreen', 'orange', 'tomato', 'orchid', 'steelblue', 'gold', 'firebrick']
nC = len(COLOR)


###################################################################

    
###################################################################
    
LOAD = np.load('data/Data_Weekly.npz', allow_pickle=True)

org_values = np.vstack([ LOAD['x'], LOAD['tx'] ])

TEST = ['Hemoglobin', 'Leukocytter', 'Sodium', 'CRP', 'Kalium', 'Albumin', 'Kreatinin', 'Trombocytter', 'ALAT', 'Bilirubin total', 'ASAT', 'ALP', 'Amylase', 'Glukose']
                
NORM = [ 
    [11.7,17],
    [4,11],
    [137,145],
    [0,4],
    [3.5,4.4],
    [34,48],
    [45,105],
    [145,390],
    [10,70],
    [5,25],
    [15,45],
    [35,400],
    [25,120],
    [4,6],
    ]

step = 21

#for i in range(14):
#    
#    plt.subplot(7,2,i+1)
#    plt.cla()
#    x = org_values[:,i].flatten()
#    x = x[np.isfinite(x)]
#    plt.hist( x, np.linspace(0,NORM[i][1]*10,step), density=True, alpha=.75 )
#    yl = plt.ylim()
#    plt.fill_between( NORM[i], 0, 100, color='gray', alpha=.75, label='Nominal range')
#    plt.ylim(yl[0],yl[1])
#    plt.title(  TEST[i] )
#    plt.yticks([])

#plt.tight_layout()



###################################################################

#plt.suptitle('Blood Samples values for '+TEST[T],fontsize=FS2)

def plot(T,sb,unit):
    plt.subplot(2,2,sb+1) 
    plt.cla() 
    
    ORG = org_values[:,T].flatten()
    ORG = ORG[np.isfinite(ORG)]
    
    xR = 300

    #xR = int((ORG.mean()+ORG.std()*3)/100)*100

    #xR = 10**int( np.log10(ORG.max()) )
    #xR = xR * ( 1 + (ORG.max() // xR ) )

    w = np.linspace(0,xR,step)[1]
    h,b = np.histogram( ORG, np.linspace(0,xR,step), density=False)
    plt.bar( b[:-1]+w/2, h, w, alpha=.75, color="steelblue")

#    X = b[:-1]+w/2
#    Y = h /h.sum()

#    def fit_lognorm(z):
#        u,s,c = z
#        return np.abs( c * 1/(X*s*np.sqrt(2*np.pi)) * np.exp( -( (np.log(X)-u)**2 / (2 * (s**2)) ) ) - Y ).max()
#        
#    def lognorm(x,u,s,c=1):
#        return c * 1/(x*s*np.sqrt(2*np.pi)) * np.exp( -( (np.log(x)-u)**2 / (2 * (s**2)) ) )

#    res = minimize( fit_lognorm, [1,1,1] , method='COBYLA', tol=1e-9, constraints=({'type': 'ineq', 'fun': lambda x:  x[0] },{'type': 'ineq', 'fun': lambda x:  x[1] }, {'type': 'ineq', 'fun': lambda x:  x[2] }, ) )

#    plt.plot( 
#        np.linspace(0,xR,step*5 )[1:], 
#        [ lognorm(x, res.x[0], res.x[1], res.x[2])*h.sum() for x in np.linspace(0,xR,step*5 )[1:] ],
#        c='orange',
#        #label='Fitted log-normal dist.' ,
#        )

    yl = plt.ylim()
    plt.fill_between( NORM[T], 0, 2000, color='gray', alpha=.2, label='Nominal range')

    plt.xlim(0,xR)
    plt.ylim(0, int(h.max()/10)*10+20 )
    plt.xticks([0,100,200,300], [0,100,200,'300'+unit], fontsize=FS3)
    plt.yticks(fontsize=FS3)
    plt.ylabel('Counts' if sb == 0 else '',fontsize=FS3)
    #plt.legend(loc=9,borderaxespad=0,ncol=2,prop={ 'size':FS3},bbox_to_anchor=(.5,-.15))
    plt.title('True distribution', fontsize=FS3)



    plt.subplot(2,2,sb+3) 
    plt.cla() 
    mR,xR = -7,7 #-20,40

    STD =  (np.log(ORG)-np.mean(np.log(NORM[T])))/np.abs( np.diff(np.log(NORM[T])) )

    xR = max( np.abs(STD.min()), np.abs( STD.max()) )
    mR = -xR
    w = np.diff(np.linspace(mR,xR,step))[1]

    h,b = np.histogram( STD, np.linspace(mR,xR,step), density=False)
    plt.bar( b[:-1]+w/2, h, w, alpha=.75, color="seagreen")

#    X = np.exp(b[:-1]+w/2)
#    Y = h/h.sum()

#    def fit_lognorm(z):
#        u,s,c = z
#        return np.abs( c * 1/(X*s*np.sqrt(2*np.pi)) * np.exp( -( (np.log(X)-u)**2 / (2 * (s**2)) ) ) - Y ).max()
#        
#    def lognorm(x,u,s,c=1):
#        return c * 1/(x*s*np.sqrt(2*np.pi)) * np.exp( -( (np.log(x)-u)**2 / (2 * (s**2)) ) )

#    res = minimize( fit_lognorm, [1,1,1] , method='COBYLA', tol=1e-9, constraints=({'type': 'ineq', 'fun': lambda x:  x[0] },{'type': 'ineq', 'fun': lambda x:  x[1] }, {'type': 'ineq', 'fun': lambda x:  x[2] }, ) )

#    plt.plot( 
#        np.linspace(mR,xR,step*5 )[1:], 
#        [ lognorm(x, res.x[0], res.x[1], res.x[2])*h.sum() for x in np.exp(np.linspace(mR,xR,step*5 )[1:]) ],
#        c='orange',
#        label='Fitted log-normal dist.' )

    plt.fill_between( [-1,1], 0, 1000, color='gray', alpha=.25, label='Nominal range')

    plt.xlim(mR,xR)
    plt.ylim(0, int(h.max()/10)*10+20 )
    plt.xticks(fontsize=FS3)
    plt.yticks(fontsize=FS3)
    plt.ylabel('Counts' if sb == 0 else '',fontsize=FS3)
    plt.title('After log-standardization', fontsize=FS3)

#plt.legend(loc=8,borderaxespad=0,ncol=2,prop={ 'size':FS3},bbox_to_anchor=(.5,-.3))

plt.figure(1,(8,6),300)

plot(2,0,'\nmmol/L')
plot(10,1,'\nU/L')

plt.suptitle( ' '*8+TEST[2]+' '*50+TEST[10], fontsize=FS2)


plt.subplots_adjust(
top=0.87,
bottom=0.075,
left=0.1,
right=0.97,
hspace=0.5,
wspace=0.19
)

plt.savefig( 'log_std' )
plt.close()

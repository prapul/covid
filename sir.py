import pandas as pd
import numpy as np

from datetime import datetime
import pandas as pd 

from scipy import optimize
from scipy import integrate

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import seaborn as sns


sns.set(style="darkgrid")

mpl.rcParams['figure.figsize'] = (16, 9)
pd.set_option('display.max_rows', 500)

df_analyse=pd.read_csv('data/processed/datetime.csv',sep=';')  
df_analyse.sort_values('date',ascending=True).head()

N0=1000000 #max susceptible population
beta=0.4   # infection spread dynamics
gamma=0.1  # recovery rate
# condition I0+S0+R0=N0
I0=df_analyse.Germany[35]
S0=N0-I0
R0=0


def SIR_model_t(SIR,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          #S*I is the 
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt


def fit_odeint(x, beta, gamma):
    '''
    helper function for the integration
    '''
    ydata = np.array(df_analyse.Germany[35:])
    t=np.arange(len(ydata))
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:,1] 
    # we only would like to get dI

def SIR_model(SIR,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          #S*I is the 
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return([dS_dt,dI_dt,dR_dt])


def SIR_fig():
    I0=df_analyse.Germany[35]
    S0=N0-I0
    R0=0
    SIR=np.array([S0,I0,R0])
    propagation_rates=pd.DataFrame(columns={'susceptible':S0,
                                            'infected':I0,
                                            'recoverd':R0})
    
    
    
    for each_t in np.arange(100):
       
        new_delta_vec=SIR_model(SIR,beta,gamma)
       
        SIR=SIR+new_delta_vec
        
        propagation_rates=propagation_rates.append({'susceptible':SIR[0],
                                                    'infected':SIR[1],
                                                    'recovered':SIR[2]}, ignore_index=True)
    ydata = np.array(df_analyse.Germany[35:])
    t=np.arange(len(ydata))
    # ensure re-initialization 
    I0=ydata[0]
    S0=N0-I0
    R0=0
    popt=[0.4,0.1]
    fit_odeint(t, *popt)
    popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
    perr = np.sqrt(np.diag(pcov))
    fitted=fit_odeint(t, *popt)
    fig, ax = plt.subplots()
    plt.semilogy(t, ydata, 'o')
    plt.semilogy(t, fitted)
    plt.title("Fit of SIR model for Germany cases")
    plt.ylabel("Population infected")
    plt.xlabel("Days")
    #plt.show()
    html_fig = mpl_to_plotly(fig)
    return html_fig                         
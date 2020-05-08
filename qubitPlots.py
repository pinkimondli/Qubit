import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Qubit import Qubit

sns.set(font_scale=1)
sns.set_style("whitegrid", {"font.family": "serif"})
plt.rcParams["figure.dpi"]=100
plt.rcParams["savefig.bbox"]='tight'
plt.rcParams["savefig.transparent"]=True

def frequency(qubit,detuning,Bfield):
    time = np.linspace(0,1e-6,10001) # in seconds

    freq = np.fft.fftfreq(time.shape[-1],d=max(time)/time.shape[-1])
    ps = np.sqrt(qubit.Rabi(time,detuning,Bfield,'S'))
    ps_fft = np.abs(np.fft.fft(ps))

    m = np.argmax(ps_fft[20:-time.shape[-1]//2])+20

    return freq[m]

def plotFieldFreq(qubit,detuning,BAbs,BUnit,so=0):
    qubit.so = so
    qubit.T2 = -1

    qFreq=np.array([])
    omega = np.array([])
    for Bi in BAbs:
        print(Bi)
        Bfield_i = np.dot(Bi,BUnit)
        fi = frequency(qubit,detuning,Bfield_i)
        de = qubit.eigen(detuning,Bfield_i)[0]
        de = de[1]-de[0]

        omega = np.append(omega,de/qubit.hbar/(2*np.pi))
        qFreq = np.append(qFreq,fi)

    fig,ax = plt.subplots(1,figsize=(5,4))
    ax.plot(BAbs,qFreq/1e6,'o',linestyle='-',color='C0',label=str(BUnit))
    ax.plot(-BAbs,qFreq/1e6,'o',linestyle='-',color='C0')
    ax.set_xlabel('B (mT)')
    ax.set_ylabel('qubit frequency (MHz)')
    ax.set_ylim(0,50)
    ax.legend()
    figname = './Plots/plotFieldFreq_'+str(BUnit)+'_SO'+str(so)+'.png'
    fig.savefig(figname)

    plt.show()
    return qFreq,omega

def plotRabiProj(qubit,time,detuning,BAbs,BUnit):
    fig,ax = plt.subplots(1,figsize=(10,4))
    for projection in ['S','T0','TM','TP']:
        xlabel,ylabel = qubit.RabiTimePlot(time,detuning,np.dot(BAbs,BUnit),ax,projection)
    ax.legend(ncol=4,loc='upper center')
    ax.set_ylim(0,1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability')

    figname = './Plots/plotRabiProj_det_'+str(detuning)+'_field_'+str(BAbs)+str(BUnit)+'.png'
    fig.savefig(figname)
    plt.show()

def plotFunnelProj(qubit,t,d,B):
    projection = ['S','T0','TM']
    theta = [0,90]
    fig,ax = plt.subplots(3,2,figsize=(8,15))
    for i in range(2):
        th = theta[i]
        for j in range(3):
            proj = projection[j]
            d,Bfield_sym,pS_fin_sym = qubit.Funnel(t,d,B,th,proj)
            ax[j,i].pcolormesh(d*1e6,Bfield_sym*1e3,pS_fin_sym,vmin=0,vmax=1)
            ax[j,i].title.set_text(str(proj))
            ax[j,i].set_xlabel('detuning (ueV)')
            ax[j,i].set_ylabel('Bfield (mT)')
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    figname = './Plots/plotFunnelProj_t'+str(t)+'_so_'+str(qubit.so)+'.png'
    fig.savefig(figname)
    plt.show()

def main():
    qubit = Qubit(g1_ll=0.4,g2_ll=0.385,g1_pp=6,g2_pp=5.1,tc=5e-6)
    qubit.so = 300e-3

    t = 200e-9
    detuning = np.linspace(-1e-3,10e-6,101)
    BAbs = np.linspace(0,5e-3,101)
    theta = 0
#    BUnit = [1,0,0]

    plotFunnelProj(qubit,t,detuning,BAbs)
#    plotRabiProj(qubit,time,detuning,BAbs,BUnit)
#    qFreq,omega = plotFieldFreq(qubit,detuning,BAbs,BUnit,300e-3)

if __name__ == '__main__':
    main()

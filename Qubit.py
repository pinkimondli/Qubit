import numpy as np
import matplotlib.pyplot as plt

class Qubit:
    def __init__(self,g1_ll,g1_pp,g2_ll,g2_pp,tc,T2=-1,so=0):
        self.g1_ll = g1_ll
        self.g2_ll = g2_ll
        self.g1_pp = g1_pp
        self.g2_pp = g2_pp
        self.tc = tc # eV
        self.muB = 58e-6 # eV/T
        self.hbar = 6.58e-16 # eV*s
        self.T2=T2
        self.so=so

    def energies(self,d,B):
        # INPUT: d=detuning, B=Bfield, c=[g1,g2,tc,muB] the physical constants, takes only scalars
        # OUTPUT: energies of the qubit at the given settings, i.e. energies of the states S, T0, TM, TP

        EZ,dEZ = self.Zeeman(B)

        E_S = -np.sqrt(d**2 + 4*self.tc**2)
        E_T0 = d
        E_TP = E_T0+EZ
        E_TM = E_T0-EZ

        return E_S,E_T0,E_TP,E_TM

    def eigen(self,d,B):
        # INPUT: d=detuning, B=Bfield, c=[g1,g2,tc,muB] the physical constants, takes only scalars
        # OUTPUT: eigenenergies of the system at the given settings, i.e. energies E0, E1, E2, E3

        E_S,E_T0,E_TP,E_TM = self.energies(d,B)

        EZ,dEZ = self.Zeeman(B)
        so = self.so*EZ

        H = [
            [E_S,   dEZ/2, so/2,  -so/2],
            [dEZ/2, E_T0,  0,     0    ],
            [so/2,  0,     E_TM,  0    ],
            [-so/2, 0,     0,     E_TP ]
        ]

        eig = np.linalg.eig(H)
        return eig

    def Rabi(self,t,d,B,project='S'):
        # INPUT: t=time, d=detuning, B=Bfield, c=[g1,g2,tc,muB] the physical constants: takes scalar or vector for t, scalars for all other quantities, preturn is {S,T0,TP,TM}
        # OUTPUT: calculate the singlet return probability after a time t and at given energy-settings

        states = {'S':np.array([1,0,0,0]), 'T0':np.array([0,1,0,0]), 'TM':np.array([0,0,1,0]), 'TP':np.array([0,0,0,1])} # qubit basis states
        psi_0 = states['S']

        # calculate the eigenenergies and -vectors at the given setting
        eig = self.eigen(d,B)

        # psi is the singlet state in the new basis
        psi = np.dot(psi_0.T,eig[1])
        rho = np.multiply(psi.reshape(-1,1),psi)

        if project in states.keys():
            psi_f = np.dot(states[project].T,eig[1])
        else:
            raise KeyError('Choose a state from "S", "T0", "TM", "TP"')

        return self._calculRabi(t,psi_f,rho,eig)

    def RabiTimePlot(self,t,d,B,ax,project='S'):
        # INPUT: parameters for Rabi, ax = axis for the plot, preturn = state to project to {S,T0,TM,TP}
        # OUPTUT: 1D-plot of Rabi oscillations versus time

        t_factor = (np.log10(max(t)))//3*3
        t_f = 1/(10**t_factor)

        d_factor = np.log10(abs(d))//3*3
        d_f = 1/(10**d_factor)

        B_factor = np.log10(np.sqrt(np.dot(B,B)))//3*3
        B_f = 1/(10**B_factor)

        ax.plot(t*t_f,self.Rabi(t,d,B,project),label='d ={:.0f} {}eV\nB = {:.0f} {}T\nProjection to {}'.format(
            d*d_f,self._prefix(d_factor),np.sqrt(np.dot(B,B))*B_f,self._prefix(B_factor),project))
        xlabel = 'time ({}s)'.format(self._prefix(t_factor))
        ylabel = 'p (singlet)'
        return (xlabel,ylabel)


    def Ramsey(self,t,d,B,t0,d0):

        S = np.array([1,0,0,0])
        eig = self.eigen(d0,B)

        psi_init = np.dot(S.T,eig[1])
        rho = np.multiply(psi_init.reshape(-1,1),psi_init)
        Ut = np.diag(np.exp(-1j*eig[0]*t0/self.hbar))
        rho = Ut@rho@Ut.conj()

        eig = self.eigen(d,B)
        psi = np.dot(S.T,eig[1]) # return vector

        return self._calculRabi(t,psi,rho,eig,project)

    def Funnel(self,t,d,B,theta,project):
        pS_fin = []

        # initialize the unit vector:
        Bunit = np.array([np.sin(theta),0,np.cos(theta)])

        # calculate Rabi for each value of t,d,B:
        for di in d:
            for Bi in B:
                Bivec = np.dot(Bi,Bunit)
                pS_fin.append(self.Rabi(t,d=di,B=Bivec,project=project))
        pS_fin = np.array(pS_fin).reshape((len(d),len(B)))

        # having checked that it is indeed symmetric in B, we save calculation speed:
        pS_fin_sym = np.concatenate((pS_fin.T[::-1], pS_fin.T))
        Bfield_sym = np.concatenate((B[::-1],-B))

        return d,Bfield_sym,pS_fin_sym

    def _calculRabi(self,t,psi,rho,eig):
        # INPUT: t=time, psi=state to project to, rho=density matrix of the system at time t=0, eig=eigenstates/energies
        # OUTPUT: ps containing the singlet return probability for each instance in t
        psi_t = []; pS=[]
        rho_tau = np.copy(rho)

        if np.array(t).shape==():
            l=1; tau=t
        else:
            l=len(t)
            tau = t[0]

        dec_matrix = np.diag([1,1,1,1])

        if self.T2>0:
            for i in range(l):
                if i>0:
                    tau = t[i]
                Ut = np.diag(np.exp(-1j*eig[0]*tau/self.hbar))
                rho_tau = Ut@rho@Ut.conj()
                rho_offD = rho_tau - dec_matrix*rho_tau
                rho_tau = rho_tau - (1-np.exp(-tau/self.T2))*rho_offD
                pS.append(np.absolute(psi@rho_tau@psi)) # singlet probability: |<psi|psi_t>|^2
                #pS.append(np.absolute(np.dot(psi_tau,psi))**2) # singlet probability: |<psi|psi_t>|^2
        else:
            for i in range(l):
                if i>0:
                    tau = t[i]
                Ut = np.diag(np.exp(-1j*eig[0]*tau/self.hbar))
                rho_tau = Ut@rho@Ut.conj()
                pS.append(np.absolute(psi@rho_tau@psi)) # singlet probability: |<psi|psi_t>|^2
                #pS.append(np.absolute(np.dot(psi_tau,psi))**2) # singlet probability: |<psi|psi_t>|^2

        return pS

    def Zeeman(self,B):
        g1 = np.diag([self.g1_ll,self.g1_ll,self.g1_pp])
        g2 = np.diag([self.g2_ll,self.g2_ll,self.g2_pp])

        if np.dot(B,B) == 0:
            EZ = 0
            dEZ = 0
        else:
            EZ = (g2+g1)@B@B/np.sqrt(np.dot(B,B))*self.muB
            dEZ = (g2-g1)@B@B/np.sqrt(np.dot(B,B))*self.muB

        return EZ,dEZ

    def _prefix(self,factor):
        lookup = {0:'',-3:'m',-6:'u',-9:'n'}
        return lookup[int(factor)]

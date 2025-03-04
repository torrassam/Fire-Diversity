# -*- coding: utf-8 -*-
"""
Function and routines file used in biofire_main.py for MT "How fires shape biodiversity in plant communities: a study using a stochastic dynamical model" (Torrassa, 2023)
"""

import os
import json
import logging
import sys

# import numba
from random import random, uniform
import numpy as np
from numpy.random import default_rng
import scipy
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------

# (0) DYNAMIC INTEGRATION PARAMS from json file

# -------------------------------------------------------------------------------------

def settings(file_settings):
    global NP, maxT, initT, NN, N0, N1, rep, H, HY, eps, minfirerettime, firevf

    fjson = open(file_settings)
    ds = json.load(fjson)

    # community initial size
    NP = ds["algorithm"]["generation"]["n_species"]

    # SIMULATION LENGHT
    maxT = ds["algorithm"]["simulation"]["runtime"]     # years
    initT = ds["algorithm"]["simulation"]["initT"]      # years

    rep = ds["algorithm"]["simulation"]["repetition"]   # number of repetition of the simulation

    H   = ds["algorithm"]["simulation"]["delta_day"]    # d, delta t of integration LEAVE IT TO 1 DAY,DO
    eps = ds["algorithm"]["simulation"]["epsilon"]      # y^-1, minimum fire frequency when feedback
    minfirerettime = ds["algorithm"]["simulation"]["minfirerettime"]          # mean fire return time
    firevf = ds["algorithm"]["simulation"]["firevf"]

    # NUMBER OF INTEGRATION TIMESTEP (SAME AS INTEGRATION TIME IN DAYS BECAUSE h=1 d)
    NN = int(maxT * 365) #days
    N0 = int(NN/2) #days
    N1 = int(initT * 365) #days
    HY = H/365                  # y, delta t of integration

# -------------------------------------------------------------------------------------

# (2) ALGORITHM FUNCTIONS

# -------------------------------------------------------------------------------------

# (2.1) DYNAMIC FUNCTION

# -------------------------------------------------------------------------------------

# @numba.jit()
def derivs(tt, v, dv):
    """ Tilman equations system extended to NP species """
    A = np.ones(NP)         # parameters for Tilman asymetric hierarchy
    A[0] = 0.0
    ip = 0
    while ip < NP:
        dv[ip] = C[ip]*v[ip] * (1-np.sum(v[0:ip+1])) - M[ip]*v[ip] - v[ip]*np.sum((C*v)[0:ip])*A[ip]
        ip += 1

# @numba.jit()
def rk4(tt, y, dydx):
    """ Runge Kutta 4th Order """
    k2 = np.zeros(NP)
    k3 = np.zeros(NP)
    k4 = np.zeros(NP)
    h6 = HY/6.
    k1 = dydx
    y1 = y+0.5*k1*HY
    derivs(tt, y1, k2)
    y2 = y + 0.5*k2*HY
    derivs(tt, y2, k3)
    y3 = y+k3*HY
    derivs(tt, y3, k4)
    yout = y+h6*(k1+2*k2+2*k3+k4)
    return yout

# @numba.jit()
def fireocc(v, iifire):
    """ Fire Occurrence function """
    v_out = v*(1-iifire) + v*R*iifire
    return v_out


# @numba.jit()
def dyn(b, bout, steps, firevf):
    """ Complete Dynamic of the community """
    i = 0
    iifire = 0
    fv = []
    df = np.zeros(NP)
    f = np.zeros_like(b)
    
    while i < steps:
        f = b
        derivs(i, f, df)
        b = rk4(i, f, df)
        
        # stochastic fire dynamics with veg-feedback and minimum return fire of 2 yrs
        dummy = random()
        numok = round(1./(np.sum(L*b)+eps))*365
        if dummy <= 1./(numok)*H and firevf > (minfirerettime*365.):
            iifire = 1.     # FIRE
            fv.append(firevf/365.)
            firevf = 0.
        else:
            iifire = 0.     # NO FIRE
            firevf = firevf+H
        
        b = fireocc(b, iifire)      # fire occurrence function
        # set to 0.0 b < 1e-10
        # b = (1 - (b < 1e-10))*b
        bout[:,i] = b        
        # if i % 100 == 0 or iifire > 0:
        #     print(i, b, iifire)
        i += 1
    # print(bout[:,0], bout[:,-1])
    return bout, fv

#--------------- (2.2) COMMUNTY FUNCTIONS ---------------

def new_community():
    """
    Function to restart to the basic plant community - in this case, no pfts
    """
    global NP, C, M, R, L, B0ic
    NP = 0
    C = np.array([])
    M = np.array([])
    R = np.array([])
    L = np.array([])
    B0ic = np.array([[]])
    
def initial_conditions(NP, binv=0.0):
    global B0ic, spp
    # bisogna aggiungere un controllo sul numero di PFT, se fossero 100+ allora la condizione iniziale bassa andrebbe cambiata
    spp = np.zeros(NP)
    nINV = np.count_nonzero(spp)
    nNAT = len(spp) - nINV
    B0ic = np.array([])
    bmin = 0.01
    while round(1/(nNAT+1),3) <= bmin:
        bmin = bmin / 2
    AA = np.ones(nNAT)*bmin # all low
    BB = np.ones((nNAT,nNAT))*bmin + np.identity(nNAT)*(0.9-bmin*nNAT) # all low, one high
    CC = np.ones(nNAT)*round(1/(nNAT+1),3) #all highest
    B0ic = np.insert(BB, 0, AA, axis=0)
    B0ic = np.insert(B0ic, nNAT+1, CC, axis=0)
    # initial condition for invasive species
    iINV = np.nonzero(spp)          
    for i in iINV[0]:
        B0ic = np.insert(B0ic, i, np.ones(nNAT+2)*binv, axis=1)

#-------------------------------------------------------------------

def flammability(rng, frt_min=1, frt_max=1000, size=1):
    """ Generation function for flammability L trait value """
    if frt_min<=1:
        frt_min = 1 + sys.float_info.epsilon #add an epsilon to exclude the lower bound, while the upper bound is automatically excluded in the numpy.Generator.uniform function
    frt = np.power(10, rng.uniform(np.log10(frt_min), np.log10(frt_max),size)) # specific fire return time
    return 1/frt

#-------------------------------------------------------------------
# GENERATION OF LINEAR C-M BASED ON THE MEDITERRANEAN and BOREAL ECOSYSTEM
#-------------------------------------------------------------------

def trait_linear(rng, t_max=1, t_min=0.01, size=1, sigma=0.007):
    """ 
    
    returns an array of colonization rate traits
        
    Parameters
    ----------
    t_max : float, optional
        The default value is 1
    t_min : float, optional
        The default value is 0.01
    size : int or tuple of ints, optional
        Size of the desired array. Usually correspond to the size of the pft community. The default value is 1.
        
    Returns
    -------
    T : ndarray rounded to the 3rd decimal value
    """
    N = size
    # linear correlation
    ii = np.arange(0,N)+1
    alfa = (t_max-t_min)/(N-1)
    beta = t_max - alfa*N
    # print(f"N={N}, C(i) = {beta} + {alfa} * i")
    # noise
    T = alfa*ii + beta
    
    if sigma!=0:
        for i in range(0,N):
            noise = rng.normal(0, sigma)
            while (T[i]+noise)<0.001: #imposto un limite inferiore sotto il quale C non può scendere
                noise = rng.normal(0, sigma)
            T[i] += noise
    return T

def rand_linear_community_med(rng, nnew):
    global NP, C, M, R, L

    # Parametri per la generazione di C ed M presi dalla comunità del mediterraneo

    I_med = np.arange(1,7)

    C_med = np.array([0.047, 0.053, 0.045, 0.067, 0.11, 0.22])
    reg_ic = scipy.stats.linregress(x=I_med, y=C_med)
    C_min = reg_ic.slope*1 + reg_ic.intercept
    C_max = reg_ic.slope*6 + reg_ic.intercept
    C_reg = reg_ic.intercept + reg_ic.slope*I_med
    C_std = np.std(C_reg - C_med)

    M_med = np.array([0.0025, 0.008, 0.02, 0.04, 0.0667, 0.025])
    reg_im = scipy.stats.linregress(x=I_med, y=M_med)
    M_min = reg_im.slope*1 + reg_im.intercept
    M_max = reg_im.slope*6 + reg_im.intercept
    M_reg = reg_im.intercept + reg_im.slope*I_med
    M_std = np.std(M_reg - M_med)
    
    m, c = 0, 0
    while np.any(c<=m): #faccio il while solo su M in modo da non dover generare un'altra volta anche C
        c = np.round(trait_linear(rng, t_max=C_max, t_min=C_min, size=nnew, sigma=C_std), 5)
        m = np.round(trait_linear(rng, t_max=M_max, t_min=M_min, size=nnew, sigma=M_std), 5)
    
    
    r = np.round(rng.uniform(.001,1,size=nnew), 5)
    l = np.round(flammability(rng, frt_min=2, frt_max=500, size=nnew), 5)
    
    C = c
    M = m
    R = r
    L = l

    return C,M,R,L


def rand_linear_community_bor(rng, nnew):
    global NP, C, M, R, L

    # Parametri per la generazione di C ed M presi dalla comunità del mediterraneo

    I_med = np.arange(1,7)

    C_med = np.array([0.085, 0.13, 0.17])
    reg_ic = scipy.stats.linregress(x=I_med, y=C_med)
    C_min = reg_ic.slope*1 + reg_ic.intercept
    C_max = reg_ic.slope*3 + reg_ic.intercept
    C_reg = reg_ic.intercept + reg_ic.slope*I_med
    C_std = np.std(C_reg - C_med)

    M_med = np.array([0.035, 0.015, 0.023])
    reg_im = scipy.stats.linregress(x=I_med, y=M_med)
    M_min = reg_im.slope*1 + reg_im.intercept
    M_max = reg_im.slope*3 + reg_im.intercept
    M_reg = reg_im.intercept + reg_im.slope*I_med
    M_std = np.std(M_reg - M_med)
    
    m, c = 0, 0
    while np.any(c<=m): #faccio il while solo su M in modo da non dover generare un'altra volta anche C
        c = np.round(trait_linear(rng, t_max=C_max, t_min=C_min, size=nnew, sigma=C_std), 5)
        m = np.round(trait_linear(rng, t_max=M_max, t_min=M_min, size=nnew, sigma=M_std), 5)
    
    
    r = np.round(rng.uniform(.001,1,size=nnew), 5)
    l = np.round(flammability(rng, frt_min=2, frt_max=500, size=nnew), 5)

    C = c
    M = m
    R = r
    L = l

    return C,M,R,L

#--------------- (2.3) I/O FUNCTION ---------------

def eco_info():
    logging.info("Species traits:")
    inv = spp==1
    logging.info("Competition [i] - Colonization [C] - Mortality [M] - Fire Response [R] - Flammability [L] - Alien[y/n]")
    for i in range(NP):
        logging.info(f'{i+1}\t{C[i]}\t{M[i]}\t{R[i]}\t{L[i]}\t{inv[i]}')

def eco_info_file(f_info_str):
    global C, M
    with open(f_info_str,'w') as filei:
        np.savetxt(filei, list(zip(C, M, R, L, spp)), fmt='%1.5f', delimiter="\t")

def set_traits(ifile):
    global C, M, R, L, NP
    
    new_community()

    traits = np.loadtxt(ifile)
    C = traits[:,0]
    M = traits[:,1]
    R = traits[:,2]
    L = traits[:,3]
    NP = len(traits[:,0])
    
    derivs.recompile()
    rk4.recompile()
    fireocc.recompile()
    dyn.recompile()
    
    return
    
def get_traits(ifile):
    set_traits(ifile)
    c, m, r, l = C, M, R, L
    return c, m, r, l

#--------------- (2.4) FIGURE FUNCTIONS ---------------

def set_color_tab(N):

    if N<=10:
        cmap = plt.get_cmap('tab10', 10)
        my_tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    elif N<=20:
        cmap = plt.get_cmap('tab20', 20)
        my_tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    else:
        nc=int(N/10)
        mytab00 = []

        i=1
        cmap = plt.get_cmap('Blues_r', i+nc+1)  # Blue
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Oranges_r', i+nc+1)  # Orange
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Greens_r', i+nc+1)  # Green
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Reds_r', i+nc+1)  # Red
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('PRGn', i+nc*2+1)  # Purple
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('pink', i+nc+1)  # Brown
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=0
        cmap = plt.get_cmap('PiYG', i+nc*2+1)  # Pink
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Greys_r', i+nc+1)  # Grey
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=10
        cmap = plt.get_cmap('gist_stern', i+nc+1)  # Olive
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=12
        cmap = plt.get_cmap('ocean', i+nc+1)  # Cyan
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        my_tab = mytab00

    return my_tab

#------------------------------ (3) MAIN ------------------------------

if __name__ == "__main__":
    rng = default_rng()
    med_community()
    # rand_exponential_pft(rng,2,3)
    rand_exponential_invasive(rng, 1, 3)
    initial_conditions(NP)
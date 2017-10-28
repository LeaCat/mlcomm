# -*- coding: UTF-8 -*-

import numpy as np

def discrete_entr(pX):
    """
    Calculates the entropy of the discrete distribution pX,
    i.e., sum(-pX*log2(pX))
    """
    pX = np.asarray(pX, dtype=float)
    
    if abs(sum(pX) - 1) >= 1e-6:
        raise ValueError('pX must be a valid probability distribution, i.e. sum(pX) ==1.')
        
    idx = pX > 0
    H = np.sum(-pX[idx] * np.log2(pX[idx]))
    
    return H

def discrete_cross_entr(pX,pY):
    """
    Calculates the cross entropy between the distributions pX and pY,
    i.e. sum(-pX * log2(pY))
    """
    
    pX = np.asarray(pX, dtype = float)
    pY = np.asarray(pY, dtype = float)
    
    if pX.size != pY.size:
        raise ValueError('pX and pY must have the same size')
    
    if abs(sum(pX)-1) >= 1e-6:
        raise ValueError('pX must be a valid probability distribution, i.e. sum(pX) ==1.')
    else:
        pX = pX/np.sum(pX)    
    
    if abs(sum(pY)-1) >= 1e-6:
        raise ValueError('pY must be a valid probability distribution, i.e. sum(pY) ==1.')
    else:
        pY = pY/np.sum(pY)
        
    idx = pX > 0
    H = np.sum(-pX[idx] * np.log2(pY[idx]))
    
    return H

def discrete_kl_div(pX,pY):
    """
    Calculates the Kullback-Lebler divergence between the distributions pX and pY
    """
    
    D = discrete_cross_entr(pX,pY) - discrete_entr(pX)
    return D
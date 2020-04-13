#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 08:52:36 2020

@author: Yuichi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#目標：二次元正規分布を用意してプロットする
mu = np.array([0,0])#平均
sigma = np.array([[3,-2],[-2,3]])#分散共分散行列＝実対称行列


x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

X, Y = np.meshgrid(x, y)

combine=np.zeros(X.shape+(2,))
combine[:,:,0]=X
combine[:,:,1]=Y


def Gaussian(mu,sigma,combine):

    Dim=len(sigma)
    
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    
    dis=np.einsum('...k,kl,...l->...', combine-mu, sigma_inv, combine-mu)/2
    
    
    return np.exp(-dis)/(np.sqrt(2*np.pi)**Dim*sigma_det)



Z=Gaussian(mu,sigma,combine)
fig = plt.figure()
ax1= fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(X, Y, Z, rstride=3, cstride=3,cmap=cm.viridis)
ax1.view_init(55,-70)#どこから図を眺めるか
ax1.plot_surface(X, Y, Z, rstride=3, cstride=3,cmap=cm.viridis)
ax2 = fig.add_subplot(1,2,2,projection='3d')
ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
ax2.view_init(90, 270)
ax2.grid(False)
plt.show()










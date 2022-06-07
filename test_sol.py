#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 08:26:23 2022

@author: axel.nael
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np

def rotate(SE):
    y,x = SE.shape
    res = SE[y-1:0:-1,x-1:0:-1]
    return res

def coords_cle(image):
    I = image.copy()
    if len(I.shape) == 3:
        I = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
    
    d = int(I.shape[0]/168)
    I = cv2.adaptiveThreshold(I, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2*int((2*d)//2)+1, 2*int(d//4)+1)
    I = 255 - I
    
    img = I[:,:int(I.shape[1]/5)]
    
    sol = cv2.imread('Images\clefdesol2.png')
    sol = cv2.cvtColor(sol,cv2.COLOR_RGB2GRAY)
    s = 2*(int(sol.shape[1]/15)//2)+1
    sol = cv2.dilate(sol,np.ones((s,s)))
    sol = cv2.resize(sol,(int(0.45/21*I.shape[1]),int(1.4/29.7*I.shape[0])))
    sol = 255*((sol==0).astype(np.uint8))
    
    res = np.zeros(img.shape)
    y = 1
    cpt_x = 0
    cpt_y = 0    
    while y < img.shape[0] - sol.shape[0]:
        x = 0
        while x < img.shape[1] - sol.shape[1]:
            #On compte le nombre de pixels blancs en commun
            cpt = len(np.argwhere(img[y:y+sol.shape[0] , x:x+sol.shape[1]] * sol > 0))
            res[y+sol.shape[0]//2,x+sol.shape[1]//2] = cpt
            
            #Si la corrélation diminue on accélère 
            if cpt_x > cpt:
                x += 4
            #Si elle augmente on ralentit
            else:
                x += 2
            cpt_x = cpt
            cpt_y = np.amax(res[y+sol.shape[0]//2,:])
        
        #Si la corrélation diminue on accélère 
        if cpt_y > cpt:
            y += 4
        #Si elle augmente on ralentit
        else:
            y += 2
        
    
    temp = res * (res>0.9*np.amax(res))
    coords = np.argwhere(temp != 0)
    ret = []
    for y,x in coords:
        if temp[y,x] == np.amax(temp[y-5*d:y+5*d,x-5*d:x+5*d]):
            ret.append([y,x])
            
    return ret

def alterations(image,cles):
    I = image.copy()
    if len(I.shape) == 3:
        I = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
    
    d = int(I.shape[0]/168)
    I = cv2.adaptiveThreshold(I, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2*int((2*d)//2)+1, 2*int(d//4)+1)
    I = 255 - I
    
    img = I[:,:int(I.shape[1]/5)]
    
    diese = 255 - cv2.cvtColor(cv2.imread('./Images/diese.png'),cv2.COLOR_RGB2GRAY)
    diese = cv2.resize(diese,(int(I.shape[1]/105),int(I.shape[0]/59)))
    diese = (255*(diese>0)).astype(np.uint8)

    bemol = 255 - cv2.cvtColor(cv2.imread('./Images/bemol.png'),cv2.COLOR_RGB2GRAY)
    bemol = cv2.resize(bemol,(int(I.shape[1]/105),int(I.shape[0]/59)))
    bemol = (255*(bemol>0)).astype(np.uint8)
    
    res_diese = np.zeros(img.shape)
    
    for y,x in cles:
        for dy in range(-4*d,4*d):
            for dx in range(12*d):
                res_diese[y + dy,x + dx] = np.count_nonzero(I[y+dy:y+dy+diese.shape[0] , x+dx:x+dx+diese.shape[1]]*diese)
                
    return res_diese
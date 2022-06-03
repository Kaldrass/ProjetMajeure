#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:04:47 2022

@author: axel.nael
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#H : 1/59.4, L = 1/105

I = cv2.imread('./Images/im4v2.png')
I = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)

d = int(I.shape[0]/168)
I = cv2.adaptiveThreshold(I, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2*int((2*d)//2)+1, 2*int(d//4)+1)
I = 255 - I

diese = 255 - cv2.cvtColor(cv2.imread('./Images/diese.png'),cv2.COLOR_RGB2GRAY)
diese = cv2.resize(diese,(int(I.shape[1]/105),int(I.shape[0]/59)))
diese = (255*(diese>0)).astype(np.uint8)

bemol = 255 - cv2.cvtColor(cv2.imread('./Images/bemol.png'),cv2.COLOR_RGB2GRAY)
bemol = cv2.resize(bemol,(int(I.shape[1]/105),int(I.shape[0]/59)))
bemol = (255*(bemol>0)).astype(np.uint8)


d2 = int(d/5)
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.95*d),int(0.95*d)))
SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d2,d2))

I_f = cv2.morphologyEx(I, cv2.MORPH_CLOSE, SE2, iterations = 1)

I_e = cv2.erode(I_f,SE)

notes = np.argwhere(I_e == 255) #Listes des coordonnées des "notes"
notes_traitees = []

for n in notes:
    treated = False
    #On vérifie si la note a déjà été traitée
    for k in notes_traitees:
        if abs(k[0] - n[0]) < d/2 and abs(k[1] - n[1]) < d/2:
            treated = True
            break
        
    V = [] #Voisinage de la note
    for k in notes:
        if abs(k[0] - n[0]) < d/2 and abs(k[1] - n[1]) < d/2:
            V.append(k)
    #Calcul du barycentre du voisinage et ajout de ce dernier à la liste des notes traitées    
    if not(treated):
        B = [int(sum([V[k][0] for k in range(len(V))])/len(V)) , int(sum([V[k][1] for k in range(len(V))])/len(V))]
        notes_traitees.append(B)
        
res_diese = np.zeros(I.shape)        
for n in notes_traitees:
    y,x = n
    case = I[y-2*d:y+2*d,x-3*d:x]
    
    for i in range(0,case.shape[0] - diese.shape[0]):
        for j in range(0,case.shape[1] - diese.shape[1]):
            res_diese[y-2*d+i,x-3*d+j] = np.count_nonzero(case[i:i+diese.shape[0],j:j+diese.shape[1]] * diese)
 
res_bemol = np.zeros(I.shape)        
for n in notes_traitees:
    y,x = n
    case = I[y-2*d:y+2*d,x-3*d:x]
    
    for i in range(0,case.shape[0] - bemol.shape[0]):
        for j in range(0,case.shape[1] - bemol.shape[1]):
            res_bemol[y-2*d+i,x-3*d+j] = np.count_nonzero(case[i:i+bemol.shape[0],j:j+bemol.shape[1]] * bemol)
              
 
plt.subplot(221)
plt.imshow(I,'gray')

plt.subplot(222)
plt.imshow(cv2.dilate(255*(res_diese>150).astype(np.uint8),diese),'gray')

plt.subplot(223)
#plt.imshow(cv2.dilate(255*(res_bemol>130).astype(np.uint8),bemol),'gray')
plt.imshow(res_bemol,'gray')
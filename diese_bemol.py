#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:04:47 2022

@author: jonathan.bouyer
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#H : 1/59.4, L = 1/105

img = cv2.imread('./Images/im4v2.jpg')

img = img[int(0.03*img.shape[0]):int(0.97*img.shape[0]) , int(0.03*img.shape[1]):int(0.97*img.shape[1])]

if (len(img.shape) == 3):
    ny,nx,nc = img.shape
    I = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

d = int(I.shape[0]/168)
I = cv2.adaptiveThreshold(I, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2*int((2*d)//2)+1, 2*int(d//4)+1)
I = 255 - I

diese = 255 - cv2.cvtColor(cv2.imread('./Images/diese.png'),cv2.COLOR_RGB2GRAY)
diese = cv2.resize(diese,(int(I.shape[1]/105),int(I.shape[0]/59)))
diese = (255*(diese>0)).astype(np.uint8)

bemol = 255 - cv2.cvtColor(cv2.imread('./Images/bemol.png'),cv2.COLOR_RGB2GRAY)
bemol = cv2.resize(bemol,(int(I.shape[1]/105),int(I.shape[0]/59)))
bemol = (255*(bemol>0)).astype(np.uint8)


d2 = int(d/8)
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.9*d),int(0.9*d)))
SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d2,d2))

I_f = cv2.morphologyEx(I, cv2.MORPH_CLOSE, SE2, iterations = 1)

I_e = cv2.erode(I_f,SE)
plt.imshow(cv2.dilate(I_e,SE),'gray')
notes = np.argwhere(I_e == 255) #Listes des coordonnées des "notes"
notes_traitees = [] #Liste des positions des notes

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
print(notes_traitees)        
res_diese = np.zeros(I.shape)        
for n in notes_traitees:
    y,x = n
    case = I[y-3*d:y+2*d,x-3*d:x]
    
    for i in range(0,case.shape[0] - diese.shape[0]):
        for j in range(0,case.shape[1] - diese.shape[1]):
            res_diese[y-3*d+i,x-3*d+j] = np.count_nonzero(case[i:i+diese.shape[0],j:j+diese.shape[1]] * diese)
            
res_bemol = np.zeros(I.shape)        
for n in notes_traitees:
    y,x = n
    case = I[y-3*d:y+2*d,x-3*d:x]
    
    for i in range(0,case.shape[0] - bemol.shape[0]):
        for j in range(0,case.shape[1] - bemol.shape[1]):
            res_bemol[y-3*d+i,x-3*d+j] = 0.7*np.count_nonzero(case[i:i+bemol.shape[0],j:j+bemol.shape[1]] * bemol) + 0.3*np.count_nonzero((255-case[i:i+bemol.shape[0],j:j+bemol.shape[1]]) * (255-bemol))
   
 
plt.subplot(221)
plt.imshow(I,'gray')

n_diese = len(np.argwhere(diese != 0))
temp_diese = (res_diese > 0.75*n_diese).astype(np.uint8)
plt.subplot(222)
# plt.imshow(cv2.dilate(255*(res_diese>635).astype(np.uint8),diese),'gray')
#plt.imshow(0.33*cv2.dilate(I_e,SE) + 0.33*I + 0.33*res_diese,'gray')
plt.imshow(cv2.dilate(temp_diese,diese),'gray')


n_bemol = 0.7*len(np.argwhere(bemol != 0)) + 0.3*len(np.argwhere(bemol == 0))
temp_bemol = (res_bemol > 0.75*n_bemol).astype(np.uint8)
plt.subplot(223)
#plt.imshow(0.5*I + 0.5*res_diese,'gray')
plt.imshow(cv2.dilate(temp_bemol,bemol),'gray')
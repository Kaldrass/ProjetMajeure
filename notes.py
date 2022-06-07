#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:24:21 2022
@author: jonathan.bouyer
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def line_evaluator(rho,theta,x):
    return (rho - x*np.cos(theta))/np.sin(theta)

def line_eval(x1,y1,x2,y2,x):
    return (x-x2)/(x1-x2)*y1 + (x1-x)/(x1-x2)*y2

def lecture(image):
    img = image.copy()
    #img = img[100:2000,125:1530]
    img = img[int(0.03*img.shape[0]):int(0.97*img.shape[0]) , int(0.03*img.shape[1]):int(0.97*img.shape[1])]
    
    if (len(img.shape) == 3):
        ny,nx,nc = img.shape
        I = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    d = int(I.shape[0]/168)
    I = cv2.adaptiveThreshold(I, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2*int((2*d)//2)+1, 2*int(d//4)+1)
    I = 255 - I

    #Positionnement des portées (Transformée de Hough)
    lines = cv2.HoughLines(I,1,np.pi/1000,int(img.shape[0]/2.5))
    s = np.sqrt(nx**2 + ny**2)
    res = img.copy()
    T = []
    R = []
    dr = d
    nb = 0
    for l in lines:
        #R.sort()
        for rho,theta in l:
            if rho > 0:
                treated = False
                #On s'assure de ne pas avoir déjà une ligne similaire en mémoire
                if len(R) > 0:
                    i = 0
                    while (i < len(R)):
                        if abs(rho - R[i]) < 0.75*dr:
                            treated = True
                        i += 1
                if not(treated):
                    R.append(rho)
                    T.append(theta)
                    nb += 1
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + s*(-b))
                    y1 = int(y0 + s*(a))
                    x2 = int(x0 - s*(-b))
                    y2 = int(y0 - s*(a))
                    
                    cv2.line(res,(x1,y1),(x2,y2),(255,0,0),1)
        
    droites = sorted(zip(R,T))
    
    D = sum([droites[k][0] for k in range(4,len(droites),5)]) - sum([droites[k][0] for k in range(0,len(droites),5)])
    d = int(D/(len(R) - len(R)//5))
    #d = int(0.95*d) #A REMETTRE OU PAS ?

    #Repérage des notes
    d2 = int(d/8)
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.9*d),int(0.9*d)))
    SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d2,d2))

    I_f = cv2.morphologyEx(I, cv2.MORPH_CLOSE, SE2, iterations = 1)

    I_e = cv2.erode(I_f,SE)
    
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
            
    #Evaluation des notes
    SE = np.ones((int(0.4*d),int(1.3*d)))
    croches = cv2.morphologyEx(I,cv2.MORPH_OPEN,SE) #Ne garde que les barres croches
    croches = cv2.dilate(croches,SE)

    tones = {}
    duration = {}
    
    for n in notes_traitees:
        x = n[1]
        y = n[0]
        
        #Hauteur
        h = 0
        minimum = max(R) + 1
        
        i = 0
        #On recherche la droite la plus proche
        while i < len(droites):
            y_portee = line_evaluator(droites[i][0],droites[i][1],x)
            if abs(y_portee - y) < abs(minimum):
                minimum = y_portee - y
                h = i
                
            i += 1
        #Si l'écart à cette droite est trop grand, on ajoute/retire un demi-ton    
        h = h%5-round(2*minimum/d)/2
            
        tones[(y,x)] = h
        
        
        #Rythme
        #On compte le nombre de pixels blancs dans le voisinage de la note sur l'image de croches
        V1 = np.count_nonzero(croches[y-8*d:y+8*d , x-3*d:x]) 
        V2 = np.count_nonzero(croches[y-8*d:y+8*d , x:x+3*d]) 
        if V1 > 3*d**2 or V2 > 3*d**2:
            res[y-10:y+10,x-10:x+10,0] = 255
            duration[(y,x)] = 0.25
        elif V1 > 0.5*d**2 or V2 > 0.5*d**2:
            res[y-10:y+10,x-10:x+10,1] = 255
            duration[(y,x)] = 0.5
        else:
            res[y-10:y+10,x-10:x+10,2] = 255
            duration[(y,x)] = 1.0
    
    
    #Ajout des dièses et des bémols
    diese = 255 - cv2.cvtColor(cv2.imread('./Images/diese.png'),cv2.COLOR_RGB2GRAY)
    diese = cv2.resize(diese,(int(I.shape[1]/105),int(I.shape[0]/59)))
    diese = (255*(diese>0)).astype(np.uint8)

    bemol = 255 - cv2.cvtColor(cv2.imread('./Images/bemol.png'),cv2.COLOR_RGB2GRAY)
    bemol = cv2.resize(bemol,(int(I.shape[1]/105),int(I.shape[0]/59)))
    bemol = (255*(bemol>0)).astype(np.uint8)

    res_diese = np.zeros(I.shape)        
    for n in notes_traitees:
        y,x = n
        case = I[y-3*d:y+2*d,x-3*d:x]
        
        for i in range(0,case.shape[0] - diese.shape[0]):
            for j in range(0,case.shape[1] - diese.shape[1]):
                res_diese[y-3*d+i,x-3*d+j] = np.count_nonzero(case[i:i+diese.shape[0],j:j+diese.shape[1]] * diese)
                
    n_diese = len(np.argwhere(diese != 0))
    res_diese = (res_diese > 0.75*n_diese).astype(np.uint8)
    dieses = np.argwhere(res_diese != 0)
    coords_dieses = []
    diese_y,diese_x = diese.shape
    for n in dieses:
        treated = False
        #On vérifie si la note a déjà été traitée
        for k in coords_dieses:
            if abs(k[0] - n[0]) < d/2 and abs(k[1] - n[1]) < d/2:
                treated = True
                break
            
        V = [] #Voisinage de la note
        for k in dieses:
            if abs(k[0] - n[0]) < d/2 and abs(k[1] - n[1]) < d/2:
                V.append(k)
        #Calcul du barycentre du voisinage et ajout de ce dernier à la liste des notes traitées    
        if not(treated):
            B = [int(sum([V[k][0] for k in range(len(V))])/len(V)), int(sum([V[k][1] for k in range(len(V))])/len(V))]
            coords_dieses.append(B)
            
            
    
    res_bemol = np.zeros(I.shape)        
    for n in notes_traitees:
        y,x = n
        case = I[y-3*d:y+2*d,x-3*d:x]
        
        for i in range(0,case.shape[0] - bemol.shape[0]):
            for j in range(0,case.shape[1] - bemol.shape[1]):
                res_bemol[y-3*d+i,x-3*d+j] = 0.7*np.count_nonzero(case[i:i+bemol.shape[0],j:j+bemol.shape[1]] * bemol) + 0.3*np.count_nonzero((255-case[i:i+bemol.shape[0],j:j+bemol.shape[1]]) * (255-bemol))
    
    n_bemol = 0.7*len(np.argwhere(bemol != 0)) + 0.3*len(np.argwhere(bemol == 0))
    res_bemol = (res_bemol > 0.75*n_bemol).astype(np.uint8)
    bemols = np.argwhere(res_bemol != 0)
    coords_bemols = []
    bemol_y,bemol_x = bemol.shape
    for n in bemols:
        treated = False
        #On vérifie si la note a déjà été traitée
        for k in coords_bemols:
            if abs(k[0] - n[0]) < d/2 and abs(k[1] - n[1]) < d/2:
                treated = True
                break
            
        V = [] #Voisinage de la note
        for k in bemols:
            if abs(k[0] - n[0]) < d/2 and abs(k[1] - n[1]) < d/2:
                V.append(k)
        #Calcul du barycentre du voisinage et ajout de ce dernier à la liste des notes traitées    
        if not(treated):
            B = [int(sum([V[k][0] for k in range(len(V))])/len(V)) , int(sum([V[k][1] for k in range(len(V))])/len(V))]
            coords_bemols.append(B)    
    print(tones)
    #Traitement
    trans = {-3:76 , -2.5:74 , -2:72 , -1.5:71 , -1:69 ,-0.5:67 , 0:65 , 0.5:64 , 1:62 , 1.5:60 , 2:59 , 2.5:57 , 3:55 , 3.5:53 , 4:52 , 4.5:50 , 5:48 , 5.5:47 , 6:45 , 6.5:43 , 7:41}
    note = []
    rythme = []
    timing = []
    t = 0
    coords = list(tones.keys())
    ref = 0
    L = []
    k = 0
    alt = {coords[k]:0 for k in range(len(coords))}
    while k < len(coords):
        y,x = notes_traitees[k]
        for alt_y,alt_x in coords_dieses:
            if abs(y - alt_y - diese_y//2) < d and abs(x - alt_x - diese_x//2) < 2*d :
                alt[(y,x)] = 1
                print(y,x,"diese")
        for alt_y,alt_x in coords_bemols:
            if abs(y - alt_y - bemol_y//2) < d and abs(x - alt_x - bemol_x//2) < 2*d :
                alt[(y,x)] = -1
                print(y,x,"bemol")
                
        if abs(coords[k][0] - coords[ref][0]) < 14*d:
            L.append((coords[k][::-1]))
            k += 1
        else:
            ref = k
            L.sort()
            
            for i in range(len(L)):
                rythme.append(duration[L[i][::-1]])
                #Evitement des clés de sols considérées comme des notes
                try:
                    hauteur = trans[tones[L[i][::-1]]] + alt[L[i][::-1]]
                    note.append(hauteur)
                except:
                    continue
                timing.append(t)
                #On vérifie si 2 notes ne sont pas jouées en même temps
                if i < len(L)-1 and abs(L[i][0] - L[i+1][0]) > d:
                    t += duration[L[i][::-1]]
                elif i == len(L) - 1:
                    t += duration[L[i][::-1]]
                
            L = []
    L.sort()
    for i in range(len(L)):
        rythme.append(duration[L[i][::-1]])
        #Evitement des clés de sols considérées comme des notes
        try:
            hauteur = trans[tones[L[i][::-1]]] + alt[L[i][::-1]]
            note.append(hauteur)
        except:
            continue
        timing.append(t)
        #On vérifie si 2 notes ne sont pas jouées en même temps
        if i < len(L)-1 and abs(L[i][0] - L[i+1][0]) > d:
            t += duration[L[i][::-1]]
        elif i == len(L) - 1:
            t += 1
    
    plt.imshow(res)
    return note,rythme,timing

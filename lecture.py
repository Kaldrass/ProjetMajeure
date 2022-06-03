import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator

I = cv2.imread('Images\im4.jpg')
I = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)

sol = cv2.imread('Images\clefdesol.png')
sol = cv2.cvtColor(sol,cv2.COLOR_RGB2GRAY)

fa = cv2.imread('Images\clefdefa.png')
fa = cv2.cvtColor(fa,cv2.COLOR_RGB2GRAY)

alterations = [cv2.imread('Images\diese.png'), cv2.imread('Images\\bemol.png')]
alterations = [cv2.cvtColor(alterations[0],cv2.COLOR_RGB2GRAY), cv2.cvtColor(alterations[1],cv2.COLOR_RGB2GRAY)]

# Une clef de sol fait 1.3cm de hauteur pour 0.45cm de largeur (parfois 0.4 parfois 0.5), à faire pour la clef de fa
# Une feuille fait 21cm x 29.7cm
# La résolution de la photo varie tout le temps, donc on doit adapter la taille de l'élément structurant (la clef)

# 1 - 4.38%, 1/22,85 | 0.45/21 = 2.14%, 1/46.67
# 2 diese : 0.5/29.7 = 1.68%, 1/59.4 | 0.2/21 = 0.95%, 1/105
# 3 bemol : 0.5/29.7 = 1.68%, 1/59.4 | 0.2/21 = 0.95%, 1/105 -> mêmes dimensions que le dièse

def threshold(I,seuil):
    return cv2.threshold(I,seuil,255,cv2.THRESH_BINARY)[1]

def hotsu(I,size,C): #size doit être impair et au moins 3
    return cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,size,C)

def dilate(I,x,y):
    return cv2.dilate(I,np.ones((y,x)))

def erode(I,x,y):
    return cv2.erode(I,np.ones((x,y)))

def bottomhat(I):
    return cv2.morphologyEx(I,cv2.MORPH_BLACKHAT,np.ones((11,11)))

def gradient(I,x,y):
    return cv2.morphologyEx(I,cv2.MORPH_GRADIENT,np.ones((y,x)))

def skeletonization(I):
    skel = np.zeros(I.shape,np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while True:
        eroded = cv2.erode(I,cross)
        temp = cv2.dilate(eroded,cross)
        temp = cv2.subtract(I,temp)
        skel = cv2.bitwise_or(skel,temp)
        I = eroded.copy()
        if cv2.countNonZero(I) == 0:
            break
    return skel 

## Détection de la clef et de l'armure
def pretraitement(I, clef, alterations):
    solkeyheight = int(I.shape[0]/22.85)
    solkeywidth = int(I.shape[1]/46.67)

    # sharpheight = flatheight
    alterationsheight = int(I.shape[0]/59.40)
    alterationswidth = int(I.shape[1]/105)
    J = hotsu(I,31,15) # 90x40 pour la clef de sol
    clef = threshold(clef, 96)
    clef = clef/255 # on binarise clef
    clef = clef.astype(np.uint8)
    print(I.shape)
    print(clef.shape)
    clef = cv2.resize(clef,(solkeywidth,solkeyheight))
    alterations = [threshold(alterations[0], 127), threshold(alterations[1], 127)]
    alterations = [alterations[0]/255, alterations[1]/255]
    alterations = [alterations[0].astype(np.uint8), alterations[1].astype(np.uint8)]
    alterations = [cv2.resize(alterations[i],(alterationswidth,alterationsheight)) for i in range(len(alterations))]
    alterations = [1 - alterations[i] for i in range(len(alterations))]
    # clef = skeletonization(1-clef)
    clef = 1 - clef

    # for k in range(len(alterations)):
        
    return J, clef, alterations

# On va comparer le nombre de pixels de la clef de sol en commun avec l'image
# Pour ce faire, on regarde le premier quart de l'image pour réduire les calculs

def detectionClef(J, clef):
    img = J[:J.shape[1],:J.shape[0]//4]
    img = img//255
    img = 1 - img
    img = img.astype(np.uint8)
    xclef = []
    yclef = []
    nbrclef = 0
    sustained = True
    remainingLoops = 0
    for i in range((img.shape[0]-clef.shape[0])//2): # On fait 1 pixel sur deux en hautuer et en largeur pour gagner en temps d'exécution
        if(sustained == False):
            remainingLoops -= 1
            if(remainingLoops == 0):
                sustained = True
            else:
                continue
        for j in range((img.shape[1]-clef.shape[1])//2):
            if(cv2.countNonZero(img[2*i:2*i+clef.shape[0], 2*j:2*j+clef.shape[1]]*clef) >= cv2.countNonZero(clef)*0.6) and sustained == True: # Si le nombre de pixels en commun est supérieur à 50% (suffisant)
                nbrclef += 1
                yclef.append(2*i)
                xclef.append(2*j)
                sustained = False
                remainingLoops = clef.shape[0]//2 # On saute un nombre de ligne équivalent à la hauteur de la clef
                break
        
    print('Clefs trouvees :',nbrclef)
    print('xsol :',xclef)
    print('ysol :',yclef)
    return nbrclef, xclef, yclef, img

def detectionArmure(J,alterations,xclef,yclef):
    img = J[:J.shape[1],:J.shape[0]//4]
    img = img//255
    img = 1 - img
    img = img.astype(np.uint8)
    xarmure = []
    yarmure = []
    nbrarmure = 0
    sustained = True
    remainingLoops = 0
    for n in range(len(alterations)): #  dièse ou bémol
        for i in range(len(xclef)//2): # On fait 1 pixel sur deux en hauteur, et en largeur pour gagner en temps d'ex. 
            for j in range((img.shape[1])//4): # Largeur
                if(sustained == False):
                    remainingLoops -= 1
                if(remainingLoops == 0):
                    sustained = True
                else:
                    continue
                for k in range(len(xclef)): # On parcourt les lignes de toutes les clefs trouvées
                    if(cv2.countNonZero(img[yclef[k]+2*i:yclef[k]+2*i+alterations[n].shape[0], xclef[k]+2*j:xclef[k]+2*j+alterations[n].shape[1]]*alterations[n]) >= cv2.countNonZero(alterations[n])*0.65):
                        nbrarmure += 1
                        yarmure.append(yclef[k]+2*i)
                        xarmure.append(xclef[k]+2*j)
    print('Armures trouvees :',nbrarmure)
    print('xarmure :',xarmure)
    print('yarmure :',yarmure)
    return(nbrarmure, xarmure, yarmure)

                

J, sol, alterations = pretraitement(I, sol, alterations)
d = detectionClef(J,sol)
img = d[3]
a = detectionArmure(J,alterations,d[1],d[2])
plt.figure()
# plt.subplot(121)
# plt.imshow(alterations[0], 'gray')
# plt.subplot(122)
# plt.imshow(alterations[1], 'gray')
# plt.show()
# plt.subplot(131)
# plt.imshow(J, 'gray')
# plt.subplot(132)
# plt.imshow(alterations[0], 'gray')
# plt.subplot(133)
plt.imshow(img, 'gray')
plt.show()
plt.figure()
for i in range(a[0]):
    plt.imshow(img[a[2][i]:a[2][i]+alterations[0].shape[0], a[1][i]:a[1][i]+alterations[0].shape[1]], 'gray')
    plt.show()


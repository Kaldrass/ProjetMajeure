import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator

I = cv2.imread('Images\im3.jpg')
I = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)

sol = cv2.imread('Images\clefdesol.png')
sol = cv2.cvtColor(sol,cv2.COLOR_RGB2GRAY)

fa = cv2.imread('Images\clefdefa.png')
fa = cv2.cvtColor(fa,cv2.COLOR_RGB2GRAY)

# Une clef de sol fait 1.3cm de hauteur pour 0.45cm de largeur (parfois 0.4 parfois 0.5), à faire pour la clef de fa
# Une feuille fait 21cm x 29.7cm
# La résolution de la photo varie tout le temps, donc on doit adapter la taille de l'élément structurant (la clef)

# 1 - 4.38%, 1/22,85 | 0.45/21 -> 2.14%, 1/46.67


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
def pretraitement(I, clef):
    J = hotsu(I,31,15) # 90x40 pour la clef de sol
    clef = threshold(clef, 96)
    clef = clef/255 # on binarise clef
    clef = clef.astype(np.uint8)
    print(I.shape)
    print(clef.shape)
    clef = cv2.resize(clef,(keywidth,keyheight))
    # clef = skeletonization(1-clef)
    clef = 1 - clef
    return J, clef 

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
        for j in range((img.shape[1]-clef.shape[1])//2):
            if(cv2.countNonZero(img[2*i:2*i+clef.shape[0], 2*j:2*j+clef.shape[1]]*clef) >= cv2.countNonZero(clef)*0.5) and sustained == True: # Si le nombre de pixels en commun est supérieur à 50% (suffisant)
                nbrclef += 1
                yclef.append(2*i)
                xclef.append(2*j)
                sustained = False
                remainingLoops = clef.shape[0]//2 # On saute un nombre de ligne équivalent à la hauteur de la clef
                break
        if(sustained == False):
            remainingLoops -= 1
            if(remainingLoops == 0):
                sustained = True
    print('Clefs trouvees :',nbrclef)
    print('xsol :',xclef)
    print('ysol :',yclef)
    return nbrclef, xclef, yclef, img


keyheight = int(I.shape[0]/22.85)
keywidth = int(I.shape[1]/46.67)

J, sol = pretraitement(I, sol)
d = detectionClef(J,sol)
img = d[3]

plt.figure()
plt.subplot(131)
plt.imshow(J, 'gray')
plt.subplot(132)
plt.imshow(sol, 'gray')
plt.subplot(133)
plt.imshow(img, 'gray')
plt.show()


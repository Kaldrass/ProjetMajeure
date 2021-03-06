import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
import test_sol

I = cv2.imread('Images\im5.jpg')
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
    # clef = threshold(clef, 96)
    # clef = clef/255 # on binarise clef
    # clef = clef.astype(np.uint8)

    # prétraitement de Jonathan
    s = 2*(int(clef.shape[1]/15)//2)+1
    clef = cv2.dilate(clef,np.ones((s,s)))
    clef = cv2.resize(clef,(int(0.45/21*I.shape[1]),int(1.4/29.7*I.shape[0])))
    clef = ((clef==0).astype(np.uint8))

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
    # for i in range((img.shape[0]-clef.shape[0])//2): # On fait 1 pixel sur deux en hauteur et en largeur pour gagner en temps d'exécution
    i = 0
    while i < (img.shape[0]-clef.shape[0])//2:
        # for j in range((img.shape[1]-clef.shape[1])//2):
        j = 0
        while j < (img.shape[1]-clef.shape[1])//2:
            matchingPix = cv2.countNonZero(img[2*i:2*i+clef.shape[0], 2*j:2*j+clef.shape[1]]*clef)
            notmatchingPix = cv2.countNonZero(1-img[2*i:2*i+clef.shape[0], 2*j:2*j+clef.shape[1]]*(1-clef))
            if(matchingPix >= cv2.countNonZero(clef)*0.2 and notmatchingPix >= cv2.countNonZero(1-clef)*0.8): # Si le nombre de pixels en commun est supérieur à 50% (suffisant)
                nbrclef += 1
                yclef.append(2*i)
                xclef.append(2*j)
                i += clef.shape[0]//2 - 1 # On saute un nombre de ligne équivalent à la hauteur de la clef
                j = 0
                break
            j += 1
        i += 1
    return  xclef, yclef, img
# Fonctionne pour les dièses, mais pas encore pour les bémols, besoin de revoirl l'algorithme de détection de clefs
def detectionArmure(J,alterations,coordclef,clef):
    xclef = [k[1] for k in coordclef]
    yclef = [k[0] for k in coordclef]
    img = J[:,:max(xclef)+clef.shape[1]//2+7*alterations[0].shape[1]]
    img = img//255
    img = 1 - img
    img = img.astype(np.uint8)
    # img = erode(img, 3,1)
    xtemp = np.array([], int)
    ytemp = np.array([], int)
    xarmure = np.array([], int)
    yarmure = np.array([], int)
    nbrarmure = 0
    for n in range(len(alterations)): #  dièse ou bémol
        for k in range(len(xclef)): # On parcourt les lignes de toutes les clefs trouvées len(xclef) = nbrclefs
            i = 0            
            while i < clef.shape[0]+15: # On parcourt les lignes de la clef
                j = 0
                while j < 6*alterations[n].shape[1]: # On regarde sur une fenêtre de largeur 7*largeur de l'alteration (7 = nombre de notes max dans une armure)
                    matchingPix = cv2.countNonZero(img[yclef[k]-clef.shape[0]//2 + i -15:yclef[k]-clef.shape[0]//2 + i-15+alterations[n].shape[0], xclef[k]+clef.shape[1]//2+j:xclef[k]+clef.shape[1]//2+j+alterations[n].shape[1]]*alterations[n])
                    notmatchingPix = cv2.countNonZero((1-img[yclef[k]-clef.shape[0]//2 +i -15:yclef[k]-clef.shape[0]//2 +i-15+alterations[n].shape[0], xclef[k]+clef.shape[1]//2+j:xclef[k]+clef.shape[1]//2+j+alterations[n].shape[1]])*(1-alterations[n]))
                    if(matchingPix >= cv2.countNonZero(alterations[n])*0.7 and notmatchingPix >= cv2.countNonZero(1-alterations[n])*0.7):
                        if(xtemp.size == 0): # Si c'est la première altération trouvée (pb avec .min() d'une liste vide)
                            if(xclef[k]+clef.shape[1]//2+j <= xclef[k]+clef.shape[1]//2 + 2*alterations[n].shape[1]):
                                nbrarmure += 1
                                ytemp = np.append(ytemp,yclef[k]-clef.shape[0]//2 +i -15)
                                xtemp = np.append(xtemp,xclef[k]+clef.shape[1]//2+j)
                                j += alterations[n].shape[1]
                                i += 2
                        else:
                            if((np.absolute(xtemp - (xclef[k]+clef.shape[1]//2+j)).min() <= alterations[n].shape[1]//4 and np.absolute(ytemp - (yclef[k]-clef.shape[0]//2+i)).min() <= alterations[n].shape[0]//4)):
                                # Si on détecte deux fois la même altération
                                j+=2
                                i+=1
                                continue
                            if((np.absolute(xtemp - (xclef[k]+clef.shape[1]//2+j)).min() >= 2*alterations[n].shape[1])):
                                # Si on détecte une altération qui n'appartient pas à l'armure
                                j+=2
                                i+=1
                                continue
                            nbrarmure += 1  
                            ytemp = np.append(ytemp,yclef[k]-clef.shape[0]//2+i)
                            xtemp = np.append(xtemp,xclef[k]+clef.shape[1]//2+j)
                            # Cela correspond à la position du pixel en haut à gauche de l'altération
                            j += alterations[n].shape[1]
                            i += 2
                    j+=2
                i+=1
            xarmure = np.append(xarmure,xtemp)
            yarmure = np.append(yarmure,ytemp)
            xtemp = np.array([], int)
            ytemp = np.array([], int)
    print('Armures trouvees :',nbrarmure)
    print('xarmure :',xarmure)
    print('yarmure :',yarmure)
    return(nbrarmure, xarmure, yarmure, img)

def detectionAlterationsNotes(alterations, notes, img):
    alterationsnotes = []
    for n in range(len(notes)): # from diese_bemol.py
        for k in range(len(alterations)):
            matchingPix = cv2.countNonZero(img[notes[n][1]-alterations[k].shape[0]//2:notes[n][1]+alterations[k].shape[0]//2 + 1, notes[n][0]-2*alterations[k].shape[1]:notes[n][0]-alterations[k].shape[1]]*alterations[k])
            notmatchingPix = cv2.countNonZero(1-img[notes[n][1]-alterations[k].shape[0]//2:notes[n][1]+alterations[k].shape[0]//2 + 1, notes[n][0]-2*alterations[k].shape[1]:notes[n][0]-alterations[k].shape[1]]*(1-alterations[k]))
            if(matchingPix >= cv2.countNonZero(alterations[k])*0.7 and notmatchingPix >= cv2.countNonZero(1-alterations[k])*0.7):
                if(k == 0):
                    alterationsnotes.append(['diese',notes[n]])
                else:
                    alterationsnotes.append(['bemol',notes[n]])
    return alterationsnotes


J, sol, alterations = pretraitement(I, sol, alterations)
d = detectionClef(J,sol) # xclef, coordclef[:][0], img
coordsClef = test_sol.coords_cle(J)
print('coords Clef', coordsClef)
img = d[2]
a = detectionArmure(J,alterations,coordsClef, sol) # nbrarmure, xarmure, yarmure, img
# alterationsNotes = detectionAlterationsNotes(alterations,diese_bemol.notes,J) # alterationsnotes
nbrAlterations = a[0]//len(coordsClef) # Pas besoin d'une division entière normalement
ordreDieses = np.array(['FA','DO','SOL','RE','LA','MI','SI'])
notesAlterees = ordreDieses[:nbrAlterations]
print('notes alterees :', notesAlterees)

plt.imshow(sol, cmap='gray')   
plt.show()
plt.figure()
for i in range(a[0]):
    plt.subplot(2,a[0]/2,i+1)
    plt.imshow(img[a[2][i]:a[2][i]+alterations[0].shape[0], a[1][i]:a[1][i]+alterations[0].shape[1]], 'gray')
    plt.title('x = '+str(a[1][i])+' y = '+str(a[2][i]))
plt.show()


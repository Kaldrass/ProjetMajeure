from xml.etree.ElementInclude import XINCLUDE_FALLBACK
import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
import diese_bemol

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
    print('Clefs trouvees :', len(xclef))
    print('xsol :',xclef)
    print('ysol :',yclef)
    return  xclef, yclef, img

def detectionArmure(J,alterations,xclef,yclef,clef):
    img = J[:,:max(xclef)+clef.shape[1]+7*alterations[0].shape[1]]
    img = img//255
    img = 1 - img
    img = img.astype(np.uint8)
    # img = erode(img, 3,1)

    xarmure = np.array([], int)
    yarmure = np.array([], int)
    nbrarmure = 0

    for n in range(len(alterations)): #  dièse ou bémol
        for k in range(len(xclef)): # On parcourt les lignes de toutes les clefs trouvées len(xclef) = nbrclefs
            i = 0            
            while i < clef.shape[0]: # On parcourt les lignes de la clef
                j = 0
                while j < 6*alterations[n].shape[1]: # On regarde sur une fenêtre de largeur 7*largeur de l'alteration (7 = nombre de notes max dans une armure)
                    matchingPix = cv2.countNonZero(img[yclef[k]+i:yclef[k]+i+alterations[n].shape[0], xclef[k]+clef.shape[1]+j:xclef[k]+clef.shape[1]+j+alterations[n].shape[1]]*alterations[n])
                    notmatchingPix = cv2.countNonZero((1-img[yclef[k]+i:yclef[k]+i+alterations[n].shape[0], xclef[k]+clef.shape[1]+j:xclef[k]+clef.shape[1]+j+alterations[n].shape[1]])*(1-alterations[n]))
                    if(matchingPix >= cv2.countNonZero(alterations[n])*0.7 and notmatchingPix >= cv2.countNonZero(1-alterations[n])*0.7):
                        if(xarmure.size == 0): # Si c'est la première altération trouvée
                            nbrarmure += 1
                            yarmure = np.append(yarmure,yclef[k]+i)
                            xarmure = np.append(xarmure,xclef[k]+clef.shape[1]+j)
                            j += alterations[n].shape[1]
                            i += 2
                        else:
                            if((np.absolute(xarmure - (xclef[k]+clef.shape[1]+j)).min() <= alterations[n].shape[1]//4 and np.absolute(yarmure - (yclef[k]+i)).min() <= alterations[n].shape[0]//4)):
                                # Si on détecte deux fois la même altération
                                j+=2
                                i+=1
                                continue
                            nbrarmure += 1  
                            yarmure = np.append(yarmure,yclef[k]+i)
                            xarmure = np.append(xarmure,xclef[k]+clef.shape[1]+j)
                            # Cela correspond à la position du pixel en haut à gauche de l'altération
                            j += alterations[n].shape[1]
                            i += 2
                    j+=2
                i+=1
        

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
d = detectionClef(J,sol) # xclef, yclef, img
img = d[2]
a = detectionArmure(J,alterations,d[0],d[1], sol) # nbrarmure, xarmure, yarmure, img
# alterationsNotes = detectionAlterationsNotes(alterations,diese_bemol.notes,J) # alterationsnotes
nbrAlterations = a[0]//len(d[0]) #Pas besoin d'une division entière normalement
ordreDieses = np.array(['FA','DO','SOL','RE','LA','MI','SI'])
notesAlterees = ordreDieses[:nbrAlterations]
print(notesAlterees)
# print(alterationsNotes)
# plt.figure()
# for i in range(a[0]):
#     plt.imshow(img[a[2][i]:a[2][i]+alterations[0].shape[0], a[1][i]:a[1][i]+alterations[0].shape[1]], 'gray')
#     plt.title('x = '+str(a[1][i])+' y = '+str(a[2][i]))
#     plt.show()


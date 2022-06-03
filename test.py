import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


def intersectionBetweenLines(line1, line2):
    print(line1, line2)
    theta1, rho1 = line1
    theta2, rho2 = line2
    x0 = (rho2 * np.sin(theta1) - rho1 * np.sin(theta2)) / (np.sin(theta1) * np.cos(theta2) - np.sin(theta2) * np.cos(theta1))
    if abs(theta1) > 0.1:
        y0 = rho1/np.sin(theta1) - x0/np.tan(theta1)
    else:
        y0 = rho2/np.sin(theta2) - x0/np.tan(theta2)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

vid = cv2.VideoCapture(2)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Width x Height = ', frame_width, 'x', frame_height)

ret, frame = vid.read()
print(ret)
ny,nx,nc = frame.shape


#A4 paper sheet dimensions
l = ny - 20
L = l/1.414
  
while(ret):
    ret, frame = vid.read()
    img = copy.deepcopy(frame)
    img = img[int((ny - l)/2):int((ny + l)/2) , int((nx - L)/2):int((nx + L)/2)]
    I = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    I = cv2.filter2D(I,-1,1/25*np.ones((5,5)))
    
    #Affichage
    cv2.line(frame,(int((nx - L)/2),int((ny - l)/2)),(int((nx + L)/2),int((ny - l)/2)),(0,0,255),2)
    cv2.line(frame,(int((nx - L)/2),int((ny + l)/2)),(int((nx + L)/2),int((ny + l)/2)),(0,0,255),2)
    cv2.line(frame,(int((nx - L)/2),int((ny - l)/2)),(int((nx - L)/2),int((ny + l)/2)),(0,0,255),2)
    cv2.line(frame,(int((nx + L)/2),int((ny - l)/2)),(int((nx + L)/2),int((ny + l)/2)),(0,0,255),2)
    
    #Traitement
    #Calcul du gradient
    seuil = 2
    gy,gx = np.gradient(I)
    G = np.sqrt(gx**2 + gy**2)
    G_thr = 255*(G>seuil).astype(np.uint8)
    
    lines = cv2.HoughLines(G_thr,2,np.pi/180,300) 
    
    I2 = cv2.cvtColor(G_thr,cv2.COLOR_GRAY2RGB)
    
    dt = np.pi/180
    dr = 30
    i = 0
    h1 = False
    h2 = False
    v1 = False
    v2 = False
    T = []
    R = []
    borders = []
    s = np.sqrt(nx**2 + ny**2)

    while not(h1 and h2 and v1 and v2) and type(lines) != type(None) and i < len(lines):
        line = lines[i]
        i += 1
        for rho,theta in line:
            affichage = False
        
            if (theta < dt or theta > np.pi - dt) and abs(rho) < dr and not(v1):
                v1 = True
                affichage = True

                borders.append([theta,rho])
                
            elif (theta < dt or theta > np.pi - dt) and abs(rho) > L - dr and not(v2):
                v2 = True
                affichage = True
                
                borders.append([theta,rho])
                
            elif np.pi/2 - dt < theta < np.pi/2 + dt and rho < dr and not(h1):
                h1 = True
                affichage = True
                
                borders.append([theta,rho])
                
            elif np.pi/2 - dt < theta < np.pi/2 + dt and rho > l - dr and not(h2):
                h2 = True
                affichage = True
                
                borders.append([theta,rho])
                
            if affichage:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho + int((nx - L)/2)
                y0 = b*rho + int((ny - l)/2)
                x1 = int(x0 + s*(-b))
                y1 = int(y0 + s*(a))
                x2 = int(x0 - s*(-b))
                y2 = int(y0 - s*(a))
                cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
            
    #Is = cv2.hconcat([frame,cv2.resize(I2,(int(I2.shape[1]*frame.shape[0]/I2.shape[0]),frame.shape[0]))])        
    #cv2.imshow('frame', Is)
    
    cv2.imshow('I',frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if h1 and h2 and v1 and v2:
        break
    
    k = cv2.waitKey(100)
  
vid.release()
cv2.destroyAllWindows()

borders.sort()

[x0,y0] = intersectionBetweenLines(borders[0], borders[2])
[x1,y1] = intersectionBetweenLines(borders[0], borders[3])
[x2,y2] = intersectionBetweenLines(borders[1], borders[2])
[x3,y3] = intersectionBetweenLines(borders[1], borders[3])
# x0 += int((nx - L)/2)
# x1 += int((nx - L)/2)
# x2 += int((nx - L)/2)
# x3 += int((nx - L)/2)
# y0 += int((ny - l)/2)
# y1 += int((ny - l)/2)
# y2 += int((ny - l)/2)
# y3 += int((ny - l)/2)
# cv2.circle(frame, (x0,y0), radius=2, color=(0, 255, 0), thickness=-1)
# cv2.circle(frame, (x1,y1), radius=2, color=(0, 255, 0), thickness=-1)
# cv2.circle(frame, (x2,y2), radius=2, color=(0, 255, 0), thickness=-1)
# cv2.circle(frame, (x3,y3), radius=2, color=(0, 255, 0), thickness=-1)

# def projectionRectangle(I):
#     #On sélectionne les lignes verticales
#     seuil = 2
#     gy,gx = np.gradient(I)
#     G = np.sqrt(gx**2 + gy**2)
#     G_thr = 255*(G>seuil).astype(np.uint8)
#     lines = cv2.HoughLines(G_thr,2,np.pi/180,300)
#     #On sélectionne les lignes horizontales
#     seuil = 2
#     gx,gy = np.gradient(I)
#     G = np.sqrt(gx**2 + gy**2)
#     G_thr = 255*(G>seuil).astype(np.uint8)
#     lines2 = cv2.HoughLines(G_thr,2,np.pi/180,300)
    
#     #On cherche les intersections entre les lignes
#     intersections = []
#     for line1 in lines:
#         for line2 in lines2:
#             intersections.append(intersectionBetweenLines(line1, line2))
#     #On récupère la plus petite intersection
#     intersections = np.array(intersections)
#     min_index = np.argmin(intersections[:,0])
    
#     #On calcule les coordonnées des coins du rectangle
#     return [int(intersections[min_index,0]),int(intersections[min_index,1]),int(intersections[min_index,0]),int(intersections[min_index,1])]

# print(projectionRectangle(I))
feuille = img[min(y0,y1,y2,y3):max(y0,y1,y2,y3),min(x0,x1,x2,x3):max(x0,x1,x2,x3)]
plt.imshow(feuille)
plt.show()
print(x0,y0,x1,y1,x2,y2,x3,y3)
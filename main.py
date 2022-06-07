import cv2
import numpy as np
import matplotlib.pyplot as plt
import notes, midiv2

I = cv2.imread('Images\im7.jpg')

tones, rythm, timing = notes.lecture(I)
midiv2.ecriture_midi(tones, rythm, timing, 240, 'test.mid')
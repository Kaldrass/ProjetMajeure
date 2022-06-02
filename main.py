import cv2
import numpy as np
import matplotlib.pyplot as plt
import notes, midi
I = cv2.imread('Images\im3.jpg')

tones, rythm = notes.lecture(I)
midi.ecriture_midi(tones, rythm, 'test.mid')
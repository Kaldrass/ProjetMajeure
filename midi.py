#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:28:59 2022

@author: axel.nael
"""

import mido
import time

def ecriture_midi(tones,rythme,titre):
    """
    tones est le retour de la fonction de lecture de notres.py
    titre doit se finie en .mid
    """
    trans = {-0.5:43 , 0:41 , 0.5:40 , 1:38 , 1.5:36 , 2:35 , 2.5:33 , 3:31 , 3.5:29 , 4:28 , 4.5:26}
    #Ecriture du fichier MIDI
    mid = mido.MidiFile() #Création du fichier
    track = mido.MidiTrack() 
    mid.tracks.append(track) 
     
    note = []
    coords = list(tones.keys())
    d = 20
    ref = 0
    L = []
    k = 0
    while k < len(coords):
        print(coords[k],ref,abs(coords[k][0] - coords[ref][0]))
        if abs(coords[k][0] - coords[ref][0]) < 6*d:
            L.append((coords[k][::-1]))
            k += 1
        else:
            ref = k
            L.sort()
            print(L)
            for i in range(len(L)):
                note.append(trans[tones[L[i][::-1]]])
            L = []

    for i in range(len(L)):
        note.append(trans[tones[L[i][::-1]]])
        
    hauteur = [12]  # choix des octaves à jouer, 12 = 1 octave et 0 = original
     
    for h in hauteur:  # boucle octave à jouer par rapport aux notes d'origine
        delta = h  # nb d'octaves à ajouter ou soustraire exprimé par tranche de 12 notes
        print("   => HAUTEUR =", delta,"notes...")  # affiche nb notes en + ou - 
        
        for i in range(len(note)):  # boucle notes à jouer dans noctn (notes partition)
            track.append(mido.Message('program_change', program=64, time=0))  # n. program=instrument
            track.append(mido.Message('note_on', note = note[i] + delta, velocity = 100, time = 32))
            print("Nocturne note #", note[i],"- Durée =", (rythme[i]), "- time =", int(256 *rythme[i]))
            track.append(mido.Message('note_off', note = note[i] + delta, velocity = 67, time = int(256 *rythme[i])))
     
     
    mid.save(titre)  # enregistre le tout dans ce fichier Midi
    print("=> Fichier MIDI sauvegardé", mid, "...")  # affiche info fichier Midi
    
def lecture_midi(titre):
    """
    titre doit se finir en .mid
    """
    port1 = mido.get_output_names()
    port = mido.open_output(port1[0])
     
    
    mid = mido.MidiFile(titre)
     
    # calcul + affiche la durée de lecture du fichier Midi en h:m:s
    print("Durée de lecture =", time.strftime('%Hh:%Mm:%Ss', time.gmtime(mid.length)))
    print("Lecture en cours...")
     
    for msg in mid.play():  # boucle de lecture du fichier Midi
        port.send(msg)      # envoi fichier Midi port MidO-OUT vers IN QS-M2 Qsynth/FS
     
    port.close()
    print("Fin de lecture")
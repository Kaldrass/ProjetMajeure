#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:28:59 2022

@author: jonathan.bouyer
"""

import mido
import time

def ecriture_midi(notes,rythme,titre):
    """
    tones est le retour de la fonction de lecture de notres.py
    titre doit se finie en .mid
    """
    #Ecriture du fichier MIDI
    mid = mido.MidiFile() #Création du fichier
    track = mido.MidiTrack() 
    mid.tracks.append(track) 
        
    hauteur = [12]  # choix des octaves à jouer, 12 = 1 octave et 0 = original
     
    for h in hauteur:  # boucle octave à jouer par rapport aux notes d'origine
        delta = h  # nb d'octaves à ajouter ou soustraire exprimé par tranche de 12 notes
        print("   => HAUTEUR =", delta,"notes...")  # affiche nb notes en + ou - 
        
        for i in range(len(notes)):  # boucle notes à jouer dans noctn (notes partition)
            track.append(mido.Message('program_change', program=64, time=0))  # n. program=instrument
            track.append(mido.Message('note_on', note = notes[i] + delta, velocity = 100, time = 32))
            print("Nocturne note #", notes[i],"- Durée =", (rythme[i]), "- time =", int(256 *rythme[i]))
            track.append(mido.Message('note_off', note = notes[i] + delta, velocity = 67, time = int(256 *rythme[i])))
     
     
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
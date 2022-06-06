#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:18:10 2022

@author: jonathan.bouyer
"""

from midiutil.MidiFile import MIDIFile
import mido
import time

def ecriture_midi(notes,rythme,timing,vitesse,titre):
    # create your MIDI object
    mf = MIDIFile(1)     # only 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, vitesse)

    # add some notes
    channel = 0
    volume = 100
    
    for i in range(len(notes)):
        pitch = notes[i]
        time = timing[i]
        duration = rythme[i]
        mf.addNote(track, channel, pitch, time, duration, volume)
        
    with open(titre, 'wb') as outf:
        mf.writeFile(outf)
        
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
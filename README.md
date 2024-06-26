# Praktikum2_CV: Erkennung von Handsbewegungen 
### Introduction:
In der heutigen digitalen Welt sind intuitive Benutzeroberflächen von entscheidender Bedeutung für die Interaktion mit technologischen Geräten. Handgesten-Erkennungssysteme spielen eine Rolle bei der Schaffung der Benutzererfahrungen. Diese Technologie ermöglicht es, durch einfache Handbewegungen intuitive Aufgaben auszuführen, wie das Blättern durch Präsentationsfolien (z.B.:Präsentation starten, nach rechts oder links blättern, Pointer-Funktion, etc.).

### Vorgehensweise/ ToDo:
- Erkennung und Klassifikation der Handwebegung mit CNN, bspw. Daumen hoch, Daumen unten, etc. => Anforderungen 
- Neuer Weg: Implementierung eines Algorithmus (MediaPipe zur Echtzeit-Erkennung der Handposition und Gesten mithilfe einer Kamera).
- Gesten mit den Tastatur-Kombinationen verküpfen zum Blättern sowie Pointer der Powerpoints
- Vergleichen der Verfahren

### Idee:
- Vergleich zwischen Real-time Bilderkennung (selbstimplementiert) und Erkennung der Handbewegung (MediaPipe)
- Selbsttrainiertes Algorithmus zur Erkennung der Handbewegung

  
### Referenzen/ Inspiration:
- https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
- https://link.springer.com/article/10.1007/s11760-023-02626-8#Sec3
- https://www.youtube.com/watch?v=Ia0tjqW7xKA: mit CNN (Code in der Beschreibung verlinkt)
- https://www.kaggle.com/code/sarjit07/hand-gestures-recognition-with-opencv-and-cnn: Auch mit CNN

### Datensätze
- https://www.kaggle.com/datasets/gti-upm/leapgestrecog
- https://www.kaggle.com/datasets/roobansappani/hand-gesture-recognition
- https://www.kaggle.com/datasets/ritikagiridhar/2000-hand-gestures

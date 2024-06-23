import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Laden des trainierten Modells
model = load_model('../model/gesture_recognition_cnn.h5')

# Überprüfen der Eingabeform des Modells
input_shape = model.input_shape[1:]  # Exclude batch size
print(f"Erwartete Eingabeform des Modells: {input_shape}")

# Klassenlabel (beispielsweise)
classes = ['blank', 'fist', 'five', 'ok', 'thumbsdown', 'thumbsup']

# Verbinden Sie die Kamera (0 steht für die Standardkamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden")
    exit()

# Definition der Region of Interest (ROI)
x, y, w, h = 400, 30, 240, 240  # Beispielwerte für das Rechteck

# Hintergrund-Subtraktor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=25, detectShadows=True)

# Zähler für die gespeicherten Bilder
image_count = 0

while True:
    # Erfassen Sie den Frame
    ret, frame = cap.read()

    if not ret:
        print("Fehler: Frame konnte nicht erfasst werden")
        break

    # Zeichnen Sie das Rechteck auf dem Frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extrahieren Sie die ROI
    roi = frame[y:y+h, x:x+w]

    # Anwenden des Hintergrund-Subtraktors
    fg_mask = bg_subtractor.apply(roi)

    # Schwellenwert anwenden und Morphologieoperationen anwenden
    _, thresh = cv2.threshold(fg_mask, 27, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Vorverarbeitung des ROI-Bildes für die Vorhersage
    processed_img = cv2.resize(thresh, (224, 224))  # Resize to 224x224
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    processed_img = np.expand_dims(processed_img, axis=-1) # Add channel dimension

    # Vorhersage der Handbewegung
    prediction = model.predict(processed_img)
    label = classes[np.argmax(prediction)]

    # Anzeige des Ergebnisses auf dem Frame
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Anzeigen des Frames
    cv2.imshow('Kamerabild', frame)
    cv2.imshow('ROI', thresh)

    # Beenden, wenn die Taste 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freigeben der Kamera und Schließen aller Fenster
cap.release()
cv2.destroyAllWindows()

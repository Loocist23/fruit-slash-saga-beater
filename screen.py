import pyautogui
import cv2
import numpy as np
import time
import os

# -------------------------------
# PARAMÈTRES À ADAPTER
# -------------------------------
# Définir la région de capture (x, y, largeur, hauteur)
# Par exemple, pour capturer l'écran entier (adapté à votre configuration) :
CAPTURE_REGION = (557, 271, 803, 603)

# Intervalle entre chaque capture (en secondes)
CAPTURE_INTERVAL = 1  # vous pouvez ajuster ce délai

# Dossier où les captures seront enregistrées
OUTPUT_FOLDER = "screenshots"

# -------------------------------
# Préparation du dossier de sauvegarde
# -------------------------------
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Dossier '{OUTPUT_FOLDER}' créé.")

print("Démarrage de la capture continue des écrans...")
print("Appuyez sur 'q' dans la fenêtre d'affichage pour quitter.")

# Initialise l'indice de capture
i = 0

# Boucle de capture continue
while True:
    # Capture de la région spécifiée
    screenshot = pyautogui.screenshot(region=CAPTURE_REGION)
    # Conversion de l'image PIL en tableau NumPy et passage du format RGB à BGR (pour OpenCV)
    screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Définir le nom du fichier à enregistrer (avec un index à 5 chiffres)
    filename = os.path.join(OUTPUT_FOLDER, f"screen_{i:05d}.png")
    cv2.imwrite(filename, screenshot_np)
    print(f"Capture enregistrée : {filename}")
    
    # Affichage de l'image dans une fenêtre debug
    cv2.imshow("Screenshot", screenshot_np)
    
    # Vérifier si l'utilisateur appuie sur 'q' pour quitter
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    i += 1  # incrémenter l'index de capture
    time.sleep(CAPTURE_INTERVAL)

cv2.destroyAllWindows()

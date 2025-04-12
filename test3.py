import cv2
import torch
import pyautogui
import numpy as np
import time

# -------------------------------
# Configuration de la zone de capture et du modèle
# -------------------------------
GAME_REGION = (557, 271, 803, 603)  # Définissez la zone de jeu sur votre écran

# Charger votre modèle entraîné (modifiez le chemin vers votre modèle)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt', force_reload=True)

# Définition des couleurs pour le dessin (format BGR)
colors = {
    'fruit': (0, 255, 0),    # vert pour le fruit
    'bomb': (0, 0, 255),     # rouge pour la bomb
    'replay': (0, 165, 255)  # orange pour le replay
}

def main():
    print("Démarrage de la détection sur le jeu...")
    while True:
        # Capture de la zone de jeu
        screenshot = pyautogui.screenshot(region=GAME_REGION)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Utiliser le modèle pour l'inférence
        results = model(frame)
        # Convertir les résultats en DataFrame (YOLOv5 fournit généralement cette option)
        df = results.pandas().xyxy[0]
        
        # Parcourir les détections et dessiner les boîtes
        for _, row in df.iterrows():
            label = row['name']
            color = colors.get(label, (255, 255, 255))  # en cas d'objet inconnu, utiliser le blanc
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            confidence = row['confidence']
            # Dessiner le rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            # Ajouter le label avec le score
            cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Afficher l'image annotée dans une fenêtre de débogage
        cv2.imshow("Game Detection", frame)
        # Quitter si on appuie sur la touche ESC (code ASCII 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        time.sleep(0.05)  # Ajustez la pause si nécessaire

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

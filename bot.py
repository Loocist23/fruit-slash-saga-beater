import cv2
import torch
import pyautogui
import time
import numpy as np
from pynput.mouse import Controller, Button
import math

# -------------------------------
# Configuration
# -------------------------------
GAME_REGION = (557, 271, 803, 603)  # Zone de la fenêtre de jeu (à ajuster)
# Charger le modèle entraîné (modifiez le chemin vers votre modèle)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True)

# Couleurs pour l'affichage (BGR)
colors = {
    'fruit': (0, 255, 0),    # Vert pour les fruits
    'bomb': (0, 0, 255),     # Rouge pour les bombes
    'replay': (0, 165, 255)  # Orange pour le bouton "replay"
}

BOMB_DISTANCE_THRESHOLD = 100   # Seuil pour considérer une bombe proche (non utilisé ici, mais peut servir)
HIGH_FRUIT_THRESHOLD = 700       # Le fruit doit être suffisamment haut (coordonnée y inférieure à cette valeur)

# Paramètres pour simuler le slice
SLICE_STEPS = 3
SLICE_SLEEP_TIME = 0.005  # Temps en secondes entre chaque petit mouvement

# Contrôleur de souris avec pynput
mouse = Controller()

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Calcule la distance entre le point (px,py) et la ligne définie par (x1,y1)-(x2,y2)
    Pour la ligne segment, si la projection tombe hors segment, renvoie la distance au point le plus proche.
    """
    line_mag = math.hypot(x2 - x1, y2 - y1)
    if line_mag < 1e-5:
        return math.hypot(px - x1, py - y1)
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    if u < 0 or u > 1:
        # la projection n'est pas sur le segment, renvoie la distance minimale aux points extrêmes
        dist1 = math.hypot(px - x1, py - y1)
        dist2 = math.hypot(px - x2, py - y2)
        return min(dist1, dist2)
    else:
        # Projection sur la ligne
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        return math.hypot(px - ix, py - iy)

def simulate_slice(center, offset=30):
    """
    Simule un mouvement de slice en déplaçant la souris de manière linéaire,
    à partir d'un point de départ calculé jusqu'à un point d'arrivée, 
    en maintenant le bouton gauche enfoncé.
    """
    cx, cy = center
    # Définir les points de slice : ici, un mouvement diagonal basé sur la taille du template
    start_x = cx - offset
    start_y = cy - offset
    end_x = cx + offset
    end_y = cy + offset
    dx = (end_x - start_x) / SLICE_STEPS
    dy = (end_y - start_y) / SLICE_STEPS

    # Positionnement initial et maintien du clic
    mouse.position = (int(start_x), int(start_y))
    mouse.press(Button.left)
    for i in range(SLICE_STEPS):
        current_x = start_x + dx * i
        current_y = start_y + dy * i
        mouse.position = (int(current_x), int(current_y))
        time.sleep(SLICE_SLEEP_TIME)
    mouse.release(Button.left)
    print(f"Slice effectué sur fruit à {center}")

def simulate_click(center):
    mouse.position = (int(center[0]), int(center[1]))
    mouse.click(Button.left, 1)
    print(f"Clic effectué sur replay à {center}")

def main():
    print("Démarrage du bot Fruit Slash...")
    while True:
        x, y, w, h = GAME_REGION
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        results = model(frame)
        df = results.pandas().xyxy[0]

        fruits, bombs, replays = [], [], []

        for _, row in df.iterrows():
            label = row['name']
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            if label == 'fruit':
                fruits.append((xmin, ymin, xmax, ymax))
            elif label == 'bomb':
                bombs.append((xmin, ymin, xmax, ymax))
            elif label == 'replay':
                replays.append((xmin, ymin, xmax, ymax))

        # Si un bouton "replay" est détecté, simuler un clic
        if replays:
            for box in replays:
                rx, ry, rx2, ry2 = box
                center_replay = (x + (rx + rx2) // 2, y + (ry + ry2) // 2)
                simulate_click(center_replay)
                time.sleep(2)
            continue

        # Pour chaque fruit détecté
        for box in fruits:
            fx, fy, fx2, fy2 = box
            center_fruit = (x + (fx + fx2) // 2, y + (fy + fy2) // 2)
            if center_fruit[1] > HIGH_FRUIT_THRESHOLD:
                print(f"Fruit à {center_fruit} trop bas (seuil: {HIGH_FRUIT_THRESHOLD}), attente...")
                continue

            # Définir le chemin du slice, basé sur le centre du fruit et un offset
            slice_start = (center_fruit[0] - (w // 10), center_fruit[1] - (h // 10))
            slice_end = (center_fruit[0] + (w // 10), center_fruit[1] + (h // 10))

            # Vérifier que la ligne de slice ne passe pas trop près d'une bombe
            bomb_in_path = False
            for b in bombs:
                bx, by, bx2, by2 = b
                center_bomb = (x + (bx + bx2) // 2, y + (by + by2) // 2)
                dist_line = point_to_line_distance(center_bomb[0], center_bomb[1],
                                                   slice_start[0], slice_start[1],
                                                   slice_end[0], slice_end[1])
                # Seuil d'intersection avec la bombe pour annuler le slice
                if dist_line < 20:  # Ajustez cette valeur selon vos tests
                    bomb_in_path = True
                    break

            if bomb_in_path:
                print(f"Slice annulé pour fruit à {center_fruit} car bombe proche sur le chemin")
                continue

            simulate_slice(center_fruit)

        debug_img = np.squeeze(results.render())
        cv2.imshow("Debug - Game", debug_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.05)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

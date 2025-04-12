import cv2
import torch
import pyautogui
import time
import numpy as np
import os
from datetime import datetime
import xml.etree.ElementTree as ET

# -------------------------------
# Configuration
# -------------------------------
GAME_REGION = (557, 271, 803, 603)  # Zone de la fenêtre du jeu (à ajuster selon votre configuration)
MODEL_PATH = 'runs/train/exp3/weights/best.pt'  # Chemin vers votre modèle entraîné
OUTPUT_DIR = 'augmented_dataset'  # Dossier de sauvegarde pour les images et annotations
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger le modèle YOLOv5 personnalisé
print("Chargement du modèle...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
print("Modèle chargé.")

# -------------------------------
# Fonction de création du fichier XML d'annotation
# -------------------------------
def create_annotation_xml(filename, image_path, img_width, img_height, detections):
    """
    Crée un arbre XML avec le format Pascal VOC à partir des détections.
    
    :param filename: Nom du fichier image (ex. "20250412_221854_815325.png")
    :param image_path: Chemin complet de l'image sauvegardée
    :param img_width: Largeur de l'image (en pixels)
    :param img_height: Hauteur de l'image (en pixels)
    :param detections: Liste de dictionnaires issus de la détection (chaque dict contient xmin, ymin, xmax, ymax, name)
    :return: Un objet ElementTree correspondant à l'annotation
    """
    annotation = ET.Element('annotation')
    
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.basename(OUTPUT_DIR)
    
    filename_el = ET.SubElement(annotation, 'filename')
    filename_el.text = filename

    path_el = ET.SubElement(annotation, 'path')
    path_el.text = os.path.abspath(image_path)

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    width_el = ET.SubElement(size, 'width')
    width_el.text = str(img_width)
    height_el = ET.SubElement(size, 'height')
    height_el.text = str(img_height)
    depth_el = ET.SubElement(size, 'depth')
    depth_el.text = '3'  # Pour une image couleur

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    # Pour chaque détection, créer un élément <object>
    for det in detections:
        obj = ET.SubElement(annotation, 'object')
        name_el = ET.SubElement(obj, 'name')
        name_el.text = det['name']  # Exemple: "fruit", "bomb", "replay"
        pose_el = ET.SubElement(obj, 'pose')
        pose_el.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin_el = ET.SubElement(bndbox, 'xmin')
        xmin_el.text = str(int(det['xmin']))
        ymin_el = ET.SubElement(bndbox, 'ymin')
        ymin_el.text = str(int(det['ymin']))
        xmax_el = ET.SubElement(bndbox, 'xmax')
        xmax_el.text = str(int(det['xmax']))
        ymax_el = ET.SubElement(bndbox, 'ymax')
        ymax_el.text = str(int(det['ymax']))
    
    return ET.ElementTree(annotation)

# -------------------------------
# Boucle de capture et annotation
# -------------------------------
def main():
    print("Démarrage de la collecte d'images annotées pour augmenter le dataset...")
    capture_interval = 0.5  # Intervalle en secondes entre deux captures
    while True:
        x, y, w, h = GAME_REGION
        # Capture d'écran de la zone définie
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Exécution de l'inférence sur l'image capturée
        results = model(frame)
        # Récupérer les détections sous forme de DataFrame
        df = results.pandas().xyxy[0]
        # Convertir la DataFrame en liste de dictionnaires
        detections = df.to_dict(orient='records')

        # Générer un nom unique pour l'image et l'annotation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"{timestamp}.png"
        img_path = os.path.join(OUTPUT_DIR, img_filename)
        # Sauvegarder l'image capturée
        cv2.imwrite(img_path, frame)
        print(f"Image sauvegardée : {img_filename}")

        # Création du fichier XML d'annotation
        xml_tree = create_annotation_xml(img_filename, img_path, w, h, detections)
        xml_filename = f"{timestamp}.xml"
        xml_path = os.path.join(OUTPUT_DIR, xml_filename)
        xml_tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        print(f"Annotation XML sauvegardée : {xml_filename}")

        # Affichage pour debug
        debug_img = np.squeeze(results.render())
        cv2.imshow("Augmentation Preview", debug_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(capture_interval)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

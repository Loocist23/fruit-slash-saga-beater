import os
import xml.etree.ElementTree as ET

# Chemins des dossiers (adapté à votre structure)
annotations_dir = "annotations"
images_dir = "images"
output_dir = "labels"  # Dossier pour les fichiers textes de labels

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Liste des classes (exemple: 'fruit' et 'bombe')
classes = ['fruit', 'bomb']

def convert_annotation(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    lines = []
    for obj in root.iter('object'):
        cls = obj.find('name').text.strip()
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        xmin = float(xml_box.find('xmin').text)
        ymin = float(xml_box.find('ymin').text)
        xmax = float(xml_box.find('xmax').text)
        ymax = float(xml_box.find('ymax').text)
        # Conversion au format YOLO
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height
        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        lines.append(line)
    return lines

# Itérer sur les fichiers XML
for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith(".xml"):
        continue
    xml_path = os.path.join(annotations_dir, xml_file)
    # Supposons que le nom de l'image correspond au nom de l'annotation (ex. screen_00000.png)
    img_filename = xml_file.replace(".xml", ".png")
    img_path = os.path.join(images_dir, img_filename)
    # Utiliser OpenCV pour obtenir la taille de l'image
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image non trouvée : {img_path}")
        continue
    h, w = img.shape[:2]
    
    yolo_lines = convert_annotation(xml_path, w, h)
    # Sauvegarder dans un fichier texte avec le même nom
    txt_filename = xml_file.replace(".xml", ".txt")
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write("\n".join(yolo_lines))
    print(f"Annotation convertie pour {img_filename}")

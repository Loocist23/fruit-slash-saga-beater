import os
import xml.etree.ElementTree as ET
import cv2

# Chemins des dossiers (adapter à votre structure)
annotations_dir = "annotations"
images_dir = "images"
output_dir = "labels"  # Dossier pour les fichiers textes de labels

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Liste des classes (exemple : 'fruit' et 'bomb')
classes = ['fruit', 'bomb']

def convert_annotation(xml_file, img_width, img_height):
    """
    Convertit une annotation au format Pascal VOC (XML) en format YOLO.
    Renvoie une liste de lignes contenant le label et les coordonnées normalisées.
    """
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
        # Conversion au format YOLO (coordonnées normalisées)
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height
        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        lines.append(line)
    return lines

def find_image(base_name):
    """
    Recherche une image dans le dossier images dont le nom de base correspond à base_name.
    Teste plusieurs extensions possibles.
    """
    extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    for ext in extensions:
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# Itérer sur tous les fichiers XML dans le dossier d'annotations
for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith(".xml"):
        continue
    xml_path = os.path.join(annotations_dir, xml_file)
    base_name = os.path.splitext(xml_file)[0]
    img_path = find_image(base_name)
    if img_path is None:
        print(f"Aucune image trouvée pour {xml_file}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Erreur lors de la lecture de l'image : {img_path}")
        continue
    h, w = img.shape[:2]

    yolo_lines = convert_annotation(xml_path, w, h)
    txt_filename = base_name + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, 'w') as f:
        f.write("\n".join(yolo_lines))
    print(f"Annotation convertie pour {os.path.basename(img_path)}")

import os
import random
import shutil

# Pour assurer la reproductibilité (optionnel)
random.seed(42)

# Dossiers source (où se trouvent toutes les images et labels)
source_images_dir = "dataset/images"
source_labels_dir = "dataset/labels"

# Dossiers de destination
dest_images_train = os.path.join(source_images_dir, "train")
dest_images_val   = os.path.join(source_images_dir, "val")
dest_labels_train = os.path.join(source_labels_dir, "train")
dest_labels_val   = os.path.join(source_labels_dir, "val")

# Créer les dossiers de destination s'ils n'existent pas déjà
for folder in [dest_images_train, dest_images_val, dest_labels_train, dest_labels_val]:
    os.makedirs(folder, exist_ok=True)

# Récupérer la liste des images (ici on suppose une extension .png)
image_files = [f for f in os.listdir(source_images_dir) if f.endswith(".png")]

# Mélanger les images de façon aléatoire
random.shuffle(image_files)

# Déterminer la répartition : 80% pour l'entraînement, 20% pour la validation
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def copy_file(src_dir, dest_dir, filename):
    src_path = os.path.join(src_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy(src_path, dest_path)

# Copier les images et leurs labels pour l'entraînement
for filename in train_files:
    copy_file(source_images_dir, dest_images_train, filename)
    # On suppose que le fichier label a le même nom en remplaçant .png par .txt
    label_filename = filename.replace(".png", ".txt")
    copy_file(source_labels_dir, dest_labels_train, label_filename)

# Copier les images et leurs labels pour la validation
for filename in val_files:
    copy_file(source_images_dir, dest_images_val, filename)
    label_filename = filename.replace(".png", ".txt")
    copy_file(source_labels_dir, dest_labels_val, label_filename)

print("La répartition du dataset est terminée.")

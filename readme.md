# Fruit Slash Saga Beater - Dataset Augmentation & Training

Ce projet permet d'entraîner un modèle YOLOv5 pour détecter des **fruits** et des **bombes** dans le jeu Fruit Slash Saga.  
Il inclut des scripts pour capturer des images, annoter des objets, convertir les annotations en format YOLO, et lancer un entraînement.

---

## 📸 1. Capturer des images

Utilise le script de capture pour prendre des screenshots continus de la zone de jeu.  
Ces images seront ensuite utilisées pour l’annotation manuelle.

---

## 🏷️ 2. Annoter les images

- Annoter chaque image capturée à la main à l’aide d’un outil comme [LabelImg](https://github.com/tzutalin/labelImg).
- Les objets détectés doivent être **fruit** ou **bomb** (orthographe stricte).
- Sauvegarder les fichiers :
  - XML dans le dossier `annotations/`
  - Images originales dans le dossier `images/`

---

## 🔁 3. Convertir en format YOLO

Utilise le script `convert.py` pour convertir les fichiers `.xml` au format YOLO (`.txt`).

```bash
python convert.py
```

Cela génèrera les fichiers `.txt` dans le dossier `labels/`.

---

## 📦 4. Préparer les datasets

Organise ton dataset comme suit :

```
datasets/
├── images/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
```

Ensuite, utilise le script `prepare.py` pour filtrer les images non labellisées :

```bash
python prepare.py
```

Ce script supprimera toutes les images du dataset principal **qui n’ont pas de fichier `.txt` correspondant dans `labels/`**.

---

## 🗂️ 5. Créer le fichier `dataset.yaml`

Exemple :

```yaml
train: datasets/images
val: datasets/images

nc: 2
names: ['fruit', 'bomb']
```

---

## 🚀 6. Lancer l'entraînement

Tu peux maintenant lancer l'entraînement YOLOv5 :

```bash
python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --name fruit-slash-detector
```

---

## 🧪 Bonus : Tester ton modèle
Utilise le script de test (`bot.py`) pour charger ton meilleur modèle (`best.pt`) et voir les prédictions en direct sur le jeu.

---
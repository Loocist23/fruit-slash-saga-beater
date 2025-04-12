# Fruit Slash Saga Beater

Fruit Slash Saga Beater est un bot automatisé qui joue à un jeu inspiré de Fruit Ninja en détectant et en "tranchant" les fruits à l’aide d’un modèle de détection d’objets (YOLOv5). Le bot évite également les bombes et peut cliquer sur le bouton de "replay" pour recommencer le jeu automatiquement.

## Table des matières

- [Présentation du Projet](#présentation-du-projet)
- [Prérequis](#prérequis)
- [Installation et Configuration](#installation-et-configuration)
- [Préparation du Dataset](#préparation-du-dataset)
- [Entraînement du Modèle](#entraînement-du-modèle)
- [Utilisation du Bot](#utilisation-du-bot)
- [Dépannage](#dépannage)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Présentation du Projet

Fruit Slash Saga Beater est conçu pour :
- Capturer en temps réel une zone définie de l’écran (la zone de jeu).
- Utiliser un modèle YOLOv5 entraîné pour détecter trois classes d’objets :  
  - `fruit` : l’objet à trancher.
  - `bomb` : l’objet à éviter.
- Simuler des actions de la souris pour réaliser des mouvements "slice" (glisser-déposer avec des trajectoires aléatoires) sur les fruits, tout en évitant les bombes.
- Cliquer sur le bouton "replay" lorsqu’il est détecté.

## Prérequis

- **Matériel :**
  - Carte graphique NVIDIA compatible avec CUDA (par exemple, NVIDIA GeForce RTX 3060 ou supérieure).
  
- **Logiciel :**
  - Python 3.8 ou version ultérieure.
  - Environnement virtuel Python (par exemple, `venv` ou `conda`).
  - CUDA (ex. CUDA 11.8) et cuDNN installés pour utiliser PyTorch avec GPU.

- **Dépendances :**
  - [PyTorch](https://pytorch.org/) avec support CUDA.
  - [YOLOv5](https://github.com/ultralytics/yolov5)
  - [OpenCV](https://opencv.org/)
  - [PyAutoGUI](https://pyautogui.readthedocs.io/)
  - [Pynput](https://pypi.org/project/pynput/)
  - NumPy

## Installation et Configuration

1. **Cloner le dépôt YOLOv5 :**

   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```

2. **Créer et activer votre environnement virtuel :**

   Sous Windows :
   ```bash
   python -m venv env
   env\Scripts\activate
   ```
   
   Sous macOS/Linux :
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Installer les dépendances :**

   Avec une version de PyTorch compatible CUDA (remplacez la commande ci-dessous par celle recommandée sur le site de PyTorch pour votre configuration, par exemple pour CUDA 11.8) :
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   pip install pyautogui pynput opencv-python numpy
   ```

4. **Organiser le dataset :**

   La structure du dataset doit ressembler à ceci :

   ```
   /dataset/
      /images/
          train/
              screen_00000.png
              screen_00001.png
              ...
          val/
              screen_00100.png
              screen_00101.png
              ...
      /labels/
          train/
              screen_00000.txt
              screen_00001.txt
              ...
          val/
              screen_00100.txt
              screen_00101.txt
              ...
   ```

5. **Créer le fichier dataset.yaml :**

   ```yaml
   train: D:\Dev\fruit-slash-saga-beater\dataset\images\train
   val: D:\Dev\fruit-slash-saga-beater\dataset\images\val

   nc: 3

   names: ['fruit', 'bomb']
   ```

## Préparation du Dataset

Si vos annotations sont au format XML (Pascal VOC), utilisez un script de conversion pour générer des fichiers .txt au format YOLO.

## Entraînement du Modèle

```bash
python train.py --img 640 --batch 16 --epochs 50 --data "D:\Dev\fruit-slash-saga-beater\dataset.yaml" --weights yolov5s.pt --device 0
```

## Utilisation du Bot

```bash
python bot.py
```

## Dépannage

- **Erreur de détection ou de slice :** Ajustez les seuils `HIGH_FRUIT_THRESHOLD`, `BOMB_DISTANCE_THRESHOLD`, etc.
- **Performances et latence :** Assurez-vous que le GPU est bien utilisé avec `torch.cuda.is_available()`.
- **Affichage debug :** Fermez la fenêtre avec ESC.

## Contribuer

Les contributions sont les bienvenues !

## Licence

Ce projet est sous licence MIT.
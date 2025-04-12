#!/usr/bin/env python
import argparse
from yolov5.train import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=int, default=640, help='Taille des images (ex: 640)')
    parser.add_argument('--batch', type=int, default=16, help='Taille du batch (ex: 16)')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d’époques')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Chemin vers le fichier dataset.yaml')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Poids initiaux (modèle pré-entraîné)')
    parser.add_argument('--device', type=str, default='0', help='Appareil à utiliser (ex: "0" pour GPU0, "cpu" pour CPU)')
    
    opt = parser.parse_args()
    
    # Lancer l'entraînement en passant également le paramètre device
    run(img=opt.img,
        batch=opt.batch,
        epochs=opt.epochs,
        data=opt.data,
        weights=opt.weights,
        device=opt.device)

#!/usr/bin/env python

import cv2 as cv
import argparse
import os
import importlib
from umucv.stream import autoStream

# Cargar clasificadores desde classifiers/
def load_classifier(name):
    try:
        return importlib.import_module(f'classifiers.{name}')
    except ModuleNotFoundError:
        print(f"[ERROR] Clasificador '{name}' no encontrado.")
        exit(1)

def load_model_features(models_dir, prepare_func):
    model_features = {}
    model_images = {}
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        img = cv.imread(path)
        if img is not None:
            model_images[file] = img
            model_features[file] = prepare_func(img)
    return model_features, model_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--dev', required=False)
    args = parser.parse_args()

    if args.method == 'help':
        print('Métodos disponibles: hist, hu, orb, mediapipe, sift, hands')
        return

    clf = load_classifier(args.method)

    # Activar visualización si el clasificador lo permite (hands)
    if hasattr(clf, "set_visualization"):
        clf.set_visualization(True)

    prepare, compare = clf.prepare, clf.compare
    model_feats, model_imgs = load_model_features(args.models, prepare)

    print(f"[INFO] Modelos cargados: {list(model_feats.keys())}")
    cv.namedWindow("result")

    for key, frame in autoStream():
        query_feat = prepare(frame)
        scores = {name: compare(query_feat, feat) for name, feat in model_feats.items()}
        best = min(scores, key=scores.get)
        conf = scores[best]

        # Mostrar info del mejor match
        cv.putText(frame, f"Best match: {best} ({conf:.4f})", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mostrar top 3
        for i, (name, score) in enumerate(sorted(scores.items(), key=lambda x: x[1])[:3]):
            cv.putText(frame, f"{name}: {score:.4f}", (10, 60 + 25*i),
                       cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

        # Llamar a la visualización personalizada del mejor match si existe (SIFT)
        if hasattr(clf, 'draw_best_match') and best in model_imgs:
            clf.draw_best_match(frame, best, os.path.join(args.models, best))

        # Guardar nuevo modelo
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if cv.waitKey(1) & 0xFF == ord('m'):
            import tkinter as tk
            from tkinter.simpledialog import askstring
            from tkinter import messagebox

            snapshot = frame.copy()
            root = tk.Tk()
            root.withdraw()

            name = askstring("Nuevo modelo", "Nombre del nuevo modelo:")
            if name:
                filename = f"{name}.jpg"
                path = os.path.join(args.models, filename)
                if os.path.exists(path):
                    overwrite = messagebox.askyesno("Sobrescribir", f"¿Sobrescribir {filename}?")
                    if not overwrite:
                        root.destroy()
                        continue

                cv.imwrite(path, snapshot)
                model_feats[filename] = prepare(snapshot)
                model_imgs[filename] = snapshot
                print(f"[INFO] Modelo '{filename}' añadido.")
            root.destroy()

        cv.imshow("result", frame)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

# Analyse von Tieren mittels Keypoint Detektoren

Dies ist das Mono Repo zum Ausführen der Ausarbeitungen von der Bachelorarbeit `Implementierung und Evaluation von Ansätzen zur sensorbasierten Analyse von Tieren mittels Key-Point Detektion`
## Verzeichnisstruktur 📂

```
monorepo/
│
├── annotation_tool/
│   ├── annotation_tool.py
│   ├── augment-script.py
│   ├── create_data.py
│
├── cv/
│   ├── main.py
│   ├── processing.py
│   ├── plotting.py
│   ├── video_utils.py
│   ├── initialization.py
│   ├── inference.py
│   ├── overlay.py
│
├── hoovenet/
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   ├── best_models/
│
├── common/
│   ├── utils.py
│   ├── constants.py
│
├── data/
│   ├── datasets/
│   │   ├── labeled/
│   │   ├── raw/
│   ├── annotations.csv
│   ├── labeled_frames.txt
│
├── requirements.txt
└── README.md
```

## Anforderungen 🛠️

Bevor du loslegst, stelle sicher, dass du alle benötigten Pakete installiert hast. Die Hauptanforderungen findest du in der `requirements.txt` Datei im Stammverzeichnis dieses Repos. Installiere sie einfach mit:

```
pip install -r requirements.txt
```

Zusätzlich brauchst du mmpose. Benutze die Installations-Anleitung von [hier](https://mmpose.readthedocs.io/en/latest/installation.html) 

## Projekte 🚀

#### Wichtig: 
In die Verzeichnisse `data/dataset/raw` müssen noch die rohen Daten, des Datensatzes. Für den Datensatz Horse-10 wurde ein Script `create_data.py` erstellt, dass die ganzen Subfolder auflöst und es zu einer Ein-Ordnerstruktur macht.


### Annotation Tool ✍️

Das Annotation Tool bietet eine grafische Benutzeroberfläche, mit der du Videoframes ganz einfach mit Hufzuständen annotieren kannst.

- **Verzeichnis**: `annotation_tool/`
- **Starten**: Navigiere zum Verzeichnis `annotation_tool` und führe das Skript `annotation_tool.py` aus:

```
python annotation_tool.py
```

Die produzierten Daten können dann mithilfe des `augment-script.py` augmentiert werden so das mehr Daten und eine bessere generalisierbarkeit/ robustheit vorhanden ist.
Hier müssen aber unbedingt die Konstanten angepasst werden um auf die richtigen Verzeichnisse zu zeigen!

### Computer Vision Pipeline 📹

Die Computer Vision Pipeline verarbeitet Videos, extrahiert Keypoints und erstellt verschiedene Visualisierungen und Overlays.

- **Verzeichnis**: `cv/`
- **Starten**: Navigiere zum Verzeichnis `cv` und führe das Skript `main.py` aus:

```
python main.py
```

### Hoovenet 🧠

Hoovenet umfasst das Training und die Verwendung eines neuronalen Netzes zur Vorhersage von Hufzuständen.

- **Verzeichnis**: `hoovenet/`
- **Training starten**: Navigiere zum Verzeichnis `hoovenet` und führe das Skript `train.py` aus:

```
python train.py
```

### Gemeinsame Bibliothek 📚

Gemeinsame Funktionen und Konstanten, die von verschiedenen Projekten verwendet werden.

- **Verzeichnis**: `common/`

## Konfiguration und Anpassung ⚙️

Alle konfigurierbaren Konstanten sind in `common/constants.py` definiert. Du kannst dort Pfade, Videoeinstellungen und andere Parameter anpassen.

## Datenverzeichnis 📁

- **`data/`**: Hier findest du die Datensätze, Frames, annotierten Frames und zugehörige Dateien. Es ist wichtig in `data/dataset/raw` die un-gelabelten Bilder hereinzulegen. Das Model wurde auf dem Datensatz Horse-10 ([Huggingface](https://huggingface.co/datasets/mwmathis/Horse-30)) vortrainiert. Falls andere Bilder zum Trainieren verwendet werden, verwende nicht das Vortrainierte Model.


Diese Abhängigkeiten, bis auf MMpose sind in der `requirements.txt` Datei im Stammverzeichnis des Monorepos aufgeführt.
# Monorepo

Dieses Monorepo enthält mehrere Projekte zur Verarbeitung von Videos, Training von neuronalen Netzen und Annotation von Daten. Jedes Projekt ist in einem eigenen Unterverzeichnis organisiert.

## Verzeichnisstruktur

\`\`\`
monorepo/
│
├── annotation_tool/
│   ├── __init__.py
│   ├── annotation_tool.py
│
├── computer_vision_pipeline/
│   ├── __init__.py
│   ├── main.py
│   ├── processing.py
│   ├── plotting.py
│   ├── video_utils.py
│   ├── initialization.py
│   ├── inference.py
│   ├── overlay.py
│
├── hoovenet/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│
├── common/
│   ├── __init__.py
│   ├── utils.py
│   ├── constants.py
│
├── data/
│   ├── datasets/
│   │   ├── raw/
│   │   ├── processed/
│   ├── frames/
│   ├── labeled_frames/
│   ├── annotations.csv
│   ├── labeled_frames.txt
│
├── requirements.txt
└── README.md
\`\`\`

## Anforderungen

Stellen Sie sicher, dass Sie die erforderlichen Pakete installiert haben. Die Hauptanforderungen sind in der \`requirements.txt\` Datei im Stammverzeichnis des Monorepos angegeben. Sie können sie mit dem folgenden Befehl installieren:

\`\`\`
pip install -r requirements.txt
\`\`\`

Zusätzlich benötigen Sie PyQt5 für das Annotation Tool. Sie können es mit dem folgenden Befehl installieren:

\`\`\`
pip install PyQt5
\`\`\`

## Projekte

### Annotation Tool

Das Annotation Tool bietet eine grafische Benutzeroberfläche zur Annotation von Videoframes mit Hufzuständen.

- **Verzeichnis**: \`annotation_tool/\`
- **Starten**: Navigieren Sie zum Verzeichnis \`annotation_tool\` und führen Sie das Skript \`annotation_tool.py\` aus:

\`\`\`
python annotation_tool.py
\`\`\`

### Computer Vision Pipeline

Die Computer Vision Pipeline verarbeitet Videos, extrahiert Keypoints und erstellt verschiedene Visualisierungen und Overlays.

- **Verzeichnis**: \`computer_vision_pipeline/\`
- **Starten**: Navigieren Sie zum Verzeichnis \`computer_vision_pipeline\` und führen Sie das Skript \`main.py\` aus:

\`\`\`
python main.py
\`\`\`

### Hoovenet

Das Hoovenet-Projekt umfasst das Training und die Verwendung eines neuronalen Netzes zur Vorhersage von Hufzuständen.

- **Verzeichnis**: \`hoovenet/\`
- **Training starten**: Navigieren Sie zum Verzeichnis \`hoovenet\` und führen Sie das Skript \`train.py\` aus:

\`\`\`
python train.py
\`\`\`

### Gemeinsame Bibliothek

Gemeinsame Funktionen und Konstanten, die von verschiedenen Projekten verwendet werden.

- **Verzeichnis**: \`common/\`

## Konfiguration und Anpassung

Alle konfigurierbaren Konstanten sind in \`common/constants.py\` definiert. Sie können dort Pfade, Videoeinstellungen und andere Parameter anpassen.

## Datenverzeichnis

- **\`data/\`**: Enthält die Datensätze, Frames, annotierten Frames und zugehörige Dateien.

## Abhängigkeiten

Stellen Sie sicher, dass Sie die erforderlichen Abhängigkeiten installiert haben. Die wichtigsten Abhängigkeiten umfassen:
- OpenCV
- PyTorch
- torchvision
- numpy
- pandas
- PyQt5

Diese Abhängigkeiten sind in der \`requirements.txt\` Datei im Stammverzeichnis des Monorepos aufgeführt.

---
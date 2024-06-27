# Annotation Tool

This submodule is part of a larger monorepo designed to annotate video frames with hoof states. The tool provides a graphical interface for labeling frames and saving annotations.

## Directory Structure

```
annotation_tool/
├── annotation_tool.py
```

## Requirements

Ensure you have the required packages installed. The main requirements are specified in the `requirements.txt` at the root of the monorepo. You can install them using:

```
pip install -r ../requirements.txt
```

Additionally, you will need PyQt5 for the graphical interface. You can install it using:

```
pip install PyQt5
```

## Files and Functionality

1. **`annotation_tool.py`**: Main script for running the annotation tool. It includes the graphical interface and logic for loading frames, making predictions, and saving annotations.

## Usage

### Step-by-Step Instructions

1. **Ensure the directory structure is in place**:
    - Make sure the `data` directory and its subdirectories (`frames`, `labeled_frames`) are properly set up as expected.
    - Ensure any required weight files for the model are placed in the correct directories, as referenced by the constants in `common/constants.py`.

2. **Run the Annotation Tool**:
    - Navigate to the `annotation_tool` directory.
    - Execute the `annotation_tool.py` script:

    ```
    python annotation_tool.py
    ```

### Key Bindings

- **Z/U/I/O**: On Ground (Left Back/Right Back/Left Front/Right Front)
- **H/J/K/L**: Off Ground (Left Back/Right Back/Left Front/Right Front)
- **E**: Skip Frame
- **W**: Save Annotation
- **Q**: Quit

### Configuration and Customization

- **Constants**: All configurable constants are defined in `common/constants.py`. You can adjust paths, video settings, and other parameters there.

### Dependencies

Make sure you have the necessary dependencies installed. The key dependencies include:
- OpenCV
- PyTorch
- torchvision
- numpy
- pandas
- PyQt5

These dependencies are listed in the `requirements.txt` file at the root of the monorepo.
# Optical Chemical Structure Recognition (OCSR)

This repository contains the implementation of an Optical Chemical Structure Recognition (OCSR) system using traditional computer vision techniques, focusing on image processing and Optical Character Recognition (OCR). It enables the conversion of chemical structure images into machine-readable formats to support cheminformatics applications.

## Features

- **Image Preprocessing**: Binarization, deskewing, cropping, and resizing using OpenCV.
- **Template Matching**: Multi-scale template matching for recognizing molecular structures.
- **Corner Detection**: Harris Corner Detection for identifying molecular nodes.
- **Bond Detection and Classification**: Using the Hough Transform and Histogram of Oriented Gradients (HOG) features for detecting and classifying bonds.
- **Text Extraction**: Tesseract OCR for extracting chemical annotations.
- **Data Representation**: Outputs chemical structures in graph formats suitable for cheminformatics.

## Installation

Clone the repository:
```bash
   git clone https://github.com/Apetun/Optical-Chemical-Structure-Recognition.git
   cd Optical-Chemical-Structure-Recognition
```
## Dataset
The project uses a dataset of 360 hand-drawn molecular structure images:

- 40 images per structure, reduced to 400x300 resolution.
- Images were binarized and processed for template matching.

## Project Structure
```
Optical-Chemical-Structure-Recognition/
│
├── data/                  # Dataset and input files
├── pickles/               # Serialized outputs (detection data, templates)
├── train/                 # Training datasets for bond classification
│
├── bond_classification.py # Bond detection and classification module
├── bond_detection.py      # Detects and validates bonds between nodes
├── corner_detection.py    # Detects corner nodes in chemical structures
├── corners_groundtruth.txt# Ground truth for corner detection
├── ocr_groundtruth.txt    # Ground truth for OCR annotations
├── resize.py              # Image preprocessing script
├── template_creation.py   # Creates molecular templates
├── template_matching.py   # Matches templates for molecular structures
│
└── README.md              # Project documentation (this file)
```
The full report on the methodology used in the project can be found in the file [Optical_Chemical_Structure_Recognition.pdf](https://github.com/user-attachments/files/18053071/Optical_Chemical_Structure_Recognition.pdf).


## License
This project is licensed under the MIT License.

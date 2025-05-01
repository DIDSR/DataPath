<p align="center">
  <h1 align="center">DataPath: A Whole Slide Image Processing Tool for AI-Ready Dataset Preparation</h1>
</p>

<p align="center">
  <img src="img/DataPath_title.png">
</p>







## Getting Started

### General Information
**`DataPath`** is an open-source computational pathology toolbox designed to support researchers and regulatory scientists in the preparation and analysis of whole slide images (WSIs). Developed to streamline and standardize WSI workflows, DataPath offers modular tools for stain normalization, color harmonization, tissue registration, and data stratification. By ensuring consistency and quality across diverse histopathology datasets, it facilitates the creation of AI-ready datasets suitable for robust algorithm development, validation, and evaluation. For more information, please contact: **[seyed.kahaki@fda.hhs.gov](mailto:seyed.kahaki@fda.hhs.gov)**.

We are continuously working on this toolbox, and we welcome any contributions.

### Modules
There are several modules in this package including
1.	WSI Handler: Includes functions and classes for general WSI analysis such as reading whole slide images, extract sub region, and visualize thumbnail.
2.	Annotation Extraction: Includes several functions for extracting annotated ROIs.
3.	Patch Extraction: Assists pathologists and developers in extracting image patches from a whole slide image's region of interest.
4.	Color Normalization: Implements multiple methods (Macenko, Vahadane, Reinhard, and Histogram Matching) to normalize staining variability across slides, ensuring consistency in downstream AI analysis.
5.	WSI Tissue Registration: Provides classical feature-based registration algorithm - ORB to align serial or cross-stained tissue sections, with support for homography and similarity transforms.
6.	Stratification: Offers tools to split and visualize datasets based on metadata for balanced train/val/test splits and reproducible AI model training.  

### Information for Developers
Code Documentation: [Link](https://didsr.github.io/DataPath/index.html)
Please refer to the code documentation and email  **[seyed.kahaki@fda.hhs.gov](mailto:seyed.kahaki@fda.hhs.gov)** if you have any questions.


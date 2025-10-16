<p align="center">
  <h1 align="center">DataPath: A Whole Slide Image Processing Tool for AI-Ready Dataset Preparation</h1>
</p>

<p align="center">
  <img src="img/DataPath_title.png">
</p>


## Getting Started

### General Information
**`DataPath`** is an open-source computational pathology toolbox designed to support researchers and regulatory scientists in the preparation and analysis of whole slide images (WSIs). Developed to streamline and standardize WSI workflows, DataPath offers modular tools for stain normalization, color harmonization, tissue registration, and data stratification. By ensuring consistency and quality across diverse histopathology datasets, it facilitates the creation of AI-ready datasets suitable for robust algorithm development, validation, and evaluation. For more information, please contact: **[seyed.kahaki@fda.hhs.gov](mailto:seyed.kahaki@fda.hhs.gov)**.

---
### Modules
There are several modules in this package including
1.	WSI Handler: Includes functions and classes for general WSI analysis such as reading whole slide images, extract sub region, and visualize thumbnail.
2.	Annotation Extraction: Includes several functions for extracting annotated ROIs.
3.	Patch Extraction: Assists pathologists and developers in extracting image patches from a whole slide image's region of interest.
4.	Color Normalization: Implements multiple methods (Macenko, Vahadane, Reinhard, and Histogram Matching) to normalize staining variability across slides, ensuring consistency in downstream AI analysis.
5.	WSI Tissue Registration: Provides classical feature-based registration algorithm - ORB to align serial or cross-stained tissue sections, with support for homography and similarity transforms.
6.	Stratification: Offers tools to split and visualize datasets based on metadata for balanced train/val/test splits and reproducible AI model training.
7.	WSI Duplicate Detection: Identifying duplicate whole slide images in a given dataset to assist AI model developers in understanding the dataset for training purposes
---
### Information for Developers
Code Documentation: [Link](https://didsr.github.io/DataPath/index.html)
Please refer to the code documentation and email  **[seyed.kahaki@fda.hhs.gov](mailto:seyed.kahaki@fda.hhs.gov)** if you have any questions.

## Installation

To set up the HistoArt environment, first clone this repository and navigate to the project directory:

```bash
git clone https://github.com/DIDSR/DataPath.git
cd DataPath
```

Create a virtual environment and install dependencies from the provided `requirements.txt`:

```bash
python3 -m venv histoart_env
source datapath_env/bin/activate
pip install -r requirements.txt
```

**Tested Environment:**
- Linux (Ubuntu 22.04 LTS recommended)
- Python 3.10+

### Dependencies

Some key dependencies include:

```sh
numpy==2.1.2
opencv-python==4.11.0.86
scikit-image==0.25.2
scikit-learn==1.6.1
matplotlib==3.10.1
pyfeats==1.0.1
mahotas==1.4.18
torch==2.5.1
torchvision==0.20.1
```

(See `requirements.txt` for the full list.)

---

## Getting Started Examples

Several Jupyter notebooks and scripts are provided to quickly familiarize you with the capabilities and usage of HistoArt:

1. [WSI Handler](https://github.com/DIDSR/DataPath/blob/main/01_read_wsi.ipynb)
2. [Annotation Extraction](https://github.com/DIDSR/DataPath/blob/main/02_annotation_extraction.ipynb)
3. [Patch Extraction](https://github.com/DIDSR/DataPath/blob/main/03_patch_extraction.ipynb)
4. [Color Normalization](https://github.com/DIDSR/DataPath/blob/main/04_color_normalization.ipynb)
5. [WSI Tissue Registration](https://github.com/DIDSR/DataPath/blob/main/05_tissue_registration.ipynb)
6. [Stratification](https://github.com/DIDSR/DataPath/blob/main/06_stratification.ipynb)
7. [WSI Duplicate Detection](https://github.com/DIDSR/DataPath/blob/main/07_duplicate_detection.ipynb)

---

## How to Cite

If you utilize HistoArt in your research or applications, please cite the repository:

```bibtex
@misc{DataPath2025,
  author = {Seyed M. Kahaki, Tanviben Patel, Alexander R. Webber, Weijie Chen},
  title = {HistoArt},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/DIDSR/DataPath}},
}
```



---

## Contact and Contributions

For any inquiries, suggestions, or collaborative opportunities, please contact Seyed Kahaki either via this GitHub repo or via email (seyed.kahaki@fda.hhs.gov).

---

## Disclaimer
### About the Catalog of Regulatory Science Tools
The enclosed tool is part of the [Catalog of Regulatory Science Tools](https://cdrh-rst.fda.gov/), which provides a peer-reviewed resource for stakeholders to use where standards and qualified Medical Device Development Tools (MDDTs) do not yet exist. These tools do not replace FDA-recognized standards or MDDTs. This catalog collates a variety of regulatory science tools that the FDA’s Center for Devices and Radiological Health’s (CDRH) Office of Science and Engineering Labs (OSEL) developed. These tools use the most innovative science to support medical device development and patient access to safe and effective medical devices. If you are considering using a tool from this catalog in your marketing submissions, note that these tools have not been qualified as [Medical Device Development Tools](https://www.fda.gov/medical-devices/medical-device-development-tools-mddt) and the FDA has not evaluated the suitability of these tools within any specific context of use. You may [request feedback or meetings for medical device submissions](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program) as part of the Q-Submission Program.
For more information about the Catalog of Regulatory Science Tools, email [RST_CDRH@fda.hhs.gov](mailto:RST_CDRH@fda.hhs.gov).

## Tool Reference
•	RST Reference Number: RSTXXXX.01

•	Date of Publication: XX/XX/XXXX

•	Recommended Citation: 

```
U.S. Food and Drug Administration. (2024). DataPath: A Whole Slide Image Processing Tool for AI-Ready Dataset Preparation (RSTXXXX.01). https://cdrh-rst.fda.gov/TBD
```

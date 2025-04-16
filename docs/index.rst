.. DataPath documentation master file, created by
   sphinx-quickstart on Wed Nov  9 10:26:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
============================================================================================================

About DataPath
--------------------------------------------------------
The Whole Slide Image Processing and Machine Learning Performance Assessment Tool is a software program written in Python for analyzing whole slide images (WSIs), assisting pathologists in the assessment of machine learning (ML) results, and evaluating ML performance. 

The toolbox includes modules to:
- Generate image patches for AI/ML models
- Extract and visualize annotations from WSIs
- Generate Aperio ImageScope annotation files for pathologist validation
- Normalize stain color across images
- Register slides across scanners
- Split datasets with category/subtype-aware stratification
- Assess model performance and metrics

The Whole Slide Image Processing and Performance Assessment Tool code has been used in the following publications:

•	Kahaki, Seyed, et al. "Weakly supervised deep learning for predicting the response to hormonal treatment of women with atypical endometrial hyperplasia: a feasibility study." Medical Imaging 2023: Digital and Computational Pathology. Vol. 12471. SPIE, 2023.
•	Kahaki, Seyed, et al. "Supervised deep learning model for ROI detection of atypical endometrial hyperplasia and endometrial cancer on histopathology whole slide images for predicting hormonal treatment response." Medical Imaging 2024: Digital and Computational Pathology.
•	Kahaki, Seyed, et al. “End-to-End Deep Learning Method for Predicting Hormonal Treatment Response in Women with Atypical Endometrial Hyperplasia or Endometrial Cancer.” Journal of Medical Imaging, Journal of Medical Imaging, Under Review
•	Mariia Sidulova, et al. “Contextual unsupervised deep clustering of digital pathology dataset”, Submitted to ISBI 2024


Modules
--------------------------------------------------------
There are several modules in this package including:

	1.	WSI handler: includes functions and classes for general WSI analysis such as read whole slide images, tissue segmentation, and normalization.
	2.	Annotation Extraction: this module includes several functions for processing annotations such as annotation extraction.
	3.	Patch Extraction: which assist pathologist and developers in extracting image patches from whole slide images region of interest.
	4.      Color Normalization: Normalizes stain color using Macenko, Vahadane, Reinhard, and histogram matching methods.
	5.	Tissue Registration: Registers tissue regions between WSIs scanned from different scanners using ORB-based alignment.
	6.	Stratification: Splits datasets into train/val/test while preserving class balance across categories and subtypes.


To see a demo of the functions in this toolbox, please refer to the Jupyter Notebooks files in the root folder of this package.

	•	01_read_wsi.ipynb_

	•	02_annotation_extraction.ipynb_

	•	03_patch_extraction.ipynb_

	•	04_color_normalization.ipynb_

	•	05_tissue_registration.ipynb_

    	•	06_stratification.ipynb_
    
.. _01_read_wsi.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/01_read_wsi.ipynb

.. _02_annotation_extraction.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/02_annotation_extraction.ipynb

.. _03_patch_extraction.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/03_patch_extraction.ipynb

.. _04_color_normalization.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/04_color_normalization.ipynb

.. _05_tissue_registration.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/05_tissue_registration.ipynb

.. _06_stratification.ipynb: https://github.com/mousavikahaki/ValidPath/blob/main/06_stratification.ipynb

.. toctree::
   :hidden:

   self

.. toctree::
	:maxdepth: 3
	:titlesonly:
   
   
   installation
   inputrequirements
   WSI
   annotation
   patch
   color_normalization
   tissue_registration
   stratification



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

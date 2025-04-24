# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:50:13 2025

@author: Tanviben.Patel
"""
"""
---------------------------------------------------------------------------
Created on Apr 2, 2025

----------------------------------------------------------------------------

**Title:**        DataPath Toolbox - WSI Handler module

**Description:**  This is the WSI Handler module for the DataPath toolbox. It is includes tissue_registration class and several methods
              
**Classes:**      tissue_registration          

This module provides the tissue_registration class, which performs tissue registration and annotation transfer between Whole Slide Images (WSIs) acquired from different scanners. (e.g., Aperio and Histech).
    
---------------------------------------------------------------------------
Author: Tanviben.Patel (tanviben.patel@fda.hhs.gov) SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
Version ='1.0'
---------------------------------------------------------------------------
"""


import os
import cv2
import glob
import numpy as np
import openslide
import tiffslide
from lxml import etree
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin


class tissue_registration:
    """
    A class for performing tissue registration and annotation transfer between
    Whole Slide Images (WSIs) from different scanners (e.g., Aperio and Histech).

    This includes registration via ORB features, annotation parsing and mapping,
    and patch extraction for downstream analysis.
    """

    def __init__(self, aperio_path, histech_path, xml_path, output_base):
        """
        Initialize the registration pipeline.

        Parameters
        ----------
        aperio_path : str
            Path to the Aperio slide (.svs).
        histech_path : str
            Path to the Histech slide (.svs).
        xml_path : str
            Path to the Aperio XML annotation file.
        output_base : str
            Base directory to save output folders and visualizations.
        """
        self.aperio_path = aperio_path
        self.histech_path = histech_path
        self.xml_path = xml_path
        self.output_base = output_base

        self.histech_dir = os.path.join(output_base, "histech_patches")
        self.aperio_dir = os.path.join(output_base, "aperio_patches")
        self.vis_dir = os.path.join(output_base, "visualizations")
        os.makedirs(self.histech_dir, exist_ok=True)
        os.makedirs(self.aperio_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.aperio_slide = openslide.OpenSlide(self.aperio_path)
        self.histech_slide = tiffslide.TiffSlide(self.histech_path)

    def load_level_image(self, slide, level=2):
        """
        Load an image from a specific resolution level of a WSI.

        Parameters
        ----------
        slide : OpenSlide or TiffSlide
            Slide object to load from.
        level : int, optional
            Resolution level to load (default is 2).

        Returns
        -------
        image : ndarray
            Image array at the specified level.
        downsample : float
            Downsampling factor relative to level 0.
        """
        if level >= slide.level_count:
            raise ValueError(f"Level {level} not available. Max level: {slide.level_count - 1}")
        dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        region = slide.read_region((0, 0), level, dims).convert("RGB")
        return np.array(region), downsample

    def register_orb(self, moving_img, fixed_img):
        """
        Register two images using ORB feature detection and matching.

        Parameters
        ----------
        moving_img : ndarray
            Source image to be aligned.
        fixed_img : ndarray
            Reference image.

        Returns
        -------
        H : ndarray
            Estimated homography matrix (3x3).
        """
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(moving_img, None)
        kp2, des2 = orb.detectAndCompute(fixed_img, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def parse_xml(self):
        """
        Parse Aperio XML annotations to extract region coordinates.

        Returns
        -------
        annotations : list of ndarray
            List of polygon coordinate arrays.
        """
        tree = etree.parse(self.xml_path)
        root = tree.getroot()
        annotations = []
        for region in root.xpath('.//Region'):
            vertices = region.xpath('.//Vertex')
            coords = [(float(v.attrib['X']), float(v.attrib['Y'])) for v in vertices]
            if coords:
                annotations.append(np.array(coords, dtype=np.float32))
        return annotations

    def map_coords(self, coords, H, scale_from, scale_to):
        """
        Apply homography and scale transformation to polygon coordinates.

        Parameters
        ----------
        coords : ndarray
            Original polygon coordinates.
        H : ndarray
            Homography matrix.
        scale_from : float
            Source scale factor.
        scale_to : float
            Target scale factor.

        Returns
        -------
        coords_transformed : ndarray
            Transformed and scaled coordinates.
        """
        coords_scaled = coords / scale_from
        coords_transformed = cv2.perspectiveTransform(coords_scaled[None], H)[0]
        return coords_transformed * scale_to

    def crop_polygon(self, slide, polygon, pad=0):
        """
        Crop a rectangular region that bounds a polygon.

        Parameters
        ----------
        slide : OpenSlide or TiffSlide
            Slide to crop from.
        polygon : ndarray
            Polygon coordinates.
        pad : int, optional
            Padding around the polygon (default is 0).

        Returns
        -------
        region : PIL.Image
            Cropped image region.
        origin : tuple
            Top-left coordinate of the cropped region.
        """
        x_min = int(np.min(polygon[:, 0])) - pad
        y_min = int(np.min(polygon[:, 1])) - pad
        x_max = int(np.max(polygon[:, 0])) + pad
        y_max = int(np.max(polygon[:, 1])) + pad
        region = slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min)).convert("RGB")
        return region, (x_min, y_min)

    def show_annotated_thumbnails(self, aperio_filename="aperio_annotated_thumb.png", histech_filename="histech_registered_thumb.png"):
        """
        Display annotated thumbnails of Aperio and Histech slides side by side.

        Parameters
        ----------
        aperio_filename : str
            Filename of the Aperio annotated thumbnail.
        histech_filename : str
            Filename of the Histech annotated thumbnail.
        """
        Image.MAX_IMAGE_PIXELS = None
        aperio_path = os.path.join(self.vis_dir, aperio_filename)
        histech_path = os.path.join(self.vis_dir, histech_filename)

        aperio_img = Image.open(aperio_path)
        histech_img = Image.open(histech_path)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(aperio_img)
        plt.title("Aperio Annotated Thumbnail")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(histech_img)
        plt.title("Histech Registered Thumbnail")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def show_registered_patches(self, max_pairs=2):
        """
        Display paired tissue patches extracted from Aperio and Histech slides.

        Parameters
        ----------
        max_pairs : int
            Number of patch pairs to display.
        """
        PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100 MB

        aperio_patches = sorted(glob.glob(os.path.join(self.aperio_dir, "*.png")))
        histech_patches = sorted(glob.glob(os.path.join(self.histech_dir, "*.png")))
        num_to_show = min(max_pairs, len(aperio_patches), len(histech_patches))

        for i in range(num_to_show):
            try:
                aperio_patch = Image.open(aperio_patches[i])
                histech_patch = Image.open(histech_patches[i])
            except ValueError as e:
                if "Decompressed data too large" in str(e):
                    print(f"?? Skipping large image: {aperio_patches[i]} or {histech_patches[i]}")
                    continue
                else:
                    raise

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(aperio_patch)
            plt.title(f"Aperio Patch {i}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(histech_patch)
            plt.title(f"Histech Patch {i}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()
            
    def save_histech_xml(self, transformed_annotations, output_file):
        """
        Save transformed annotations to an XML file (Aperio-style format).
    
        Parameters
        ----------
        transformed_annotations : list of ndarray
            List of polygon coordinate arrays mapped to Histech image.
        output_file : str
            Path to save the new XML file.
        """
        root = etree.Element("Annotations")
        for i, coords in enumerate(transformed_annotations):
            annotation = etree.SubElement(root, "Annotation", Id=str(i), Name=f"Region_{i}", Type="Polygon", LineColor="255")
            region = etree.SubElement(annotation, "Regions")
            reg = etree.SubElement(region, "Region", Id="0", Type="0", Zoom="1", ImageLocation="", Length="0")
            vertices = etree.SubElement(reg, "Vertices")
            for x, y in coords:
                etree.SubElement(vertices, "Vertex", X=str(x), Y=str(y))
        tree = etree.ElementTree(root)
        tree.write(output_file, pretty_print=True, xml_declaration=True, encoding='UTF-8')


    def run(self):
        """
        Run the full tissue registration and patch extraction pipeline.

        This includes:
        - Loading thumbnails
        - Computing homography using ORB
        - Parsing XML annotations
        - Mapping annotations
        - Extracting and saving patches
        - Drawing annotated thumbnails
        - Saving visualizations
        """
        aperio_thumb, aperio_scale = self.load_level_image(self.aperio_slide)
        histech_thumb, histech_scale = self.load_level_image(self.histech_slide)
        H = self.register_orb(aperio_thumb, histech_thumb)
        annotations = self.parse_xml()
        print(f"Found {len(annotations)} annotations in XML.")
        aperio_vis = aperio_thumb.copy()
        histech_vis = histech_thumb.copy()

        for i, aperio_coords in enumerate(annotations):
            histech_coords = self.map_coords(aperio_coords, H, aperio_scale, histech_scale)

            patch_img, (x, y) = self.crop_polygon(self.histech_slide, histech_coords)
            patch_img.save(os.path.join(self.histech_dir, f"histech_patch_{i}_from_{x}_{y}.png"))

            aperio_patch, (ax, ay) = self.crop_polygon(self.aperio_slide, aperio_coords)
            angle_rad = np.arctan2(H[1, 0], H[0, 0])
            angle_deg = np.degrees(angle_rad)
            aperio_patch_rotated = aperio_patch.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
            aperio_patch_rotated.save(os.path.join(self.aperio_dir, f"aperio_patch_{i}_rotated_{int(angle_deg)}deg_from_{ax}_{ay}.png"))

            aperio_poly_thumb = (aperio_coords / aperio_scale).astype(np.int32)
            histech_poly_thumb = (histech_coords / histech_scale).astype(np.int32)

            cv2.polylines(aperio_vis, [aperio_poly_thumb], isClosed=True, color=(0, 255, 0), thickness=8)
            cv2.polylines(histech_vis, [histech_poly_thumb], isClosed=True, color=(0, 255, 0), thickness=16)
            # Collect transformed annotations
            transformed_annotations = []
            for aperio_coords in annotations:
                histech_coords = self.map_coords(aperio_coords, H, aperio_scale, histech_scale)
                transformed_annotations.append(histech_coords)
            
        # Save new XML for Histech
        xml_output_path = os.path.join(self.output_base, "histech_annotations.xml")
        self.save_histech_xml(transformed_annotations, xml_output_path)
        print(" Saved transformed annotations to:", xml_output_path)

        Image.fromarray(aperio_vis).save(os.path.join(self.vis_dir, "aperio_annotated_thumb.png"))
        Image.fromarray(histech_vis).save(os.path.join(self.vis_dir, "histech_registered_thumb.png"))

        print("Aperio slide levels:", self.aperio_slide.level_dimensions)
        print("Histech slide levels:", self.histech_slide.level_dimensions)
        print("\n Registration complete.")
        print(" Saved Aperio patches to:", self.aperio_dir)
        print(" Saved Histech patches to:", self.histech_dir)
        print(" Saved visualizations to:", self.vis_dir)

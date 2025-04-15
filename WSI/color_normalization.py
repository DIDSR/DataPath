"""
---------------------------------------------------------------------------
Created on Apr 7, 2025

----------------------------------------------------------------------------

**Title:**        DataPath Toolbox - WSI Handler module

**Description:**  This is the WSI Handler module for the DataPath toolbox. It is includes color_normalization class and several methods
              
**Classes:**      color_normalization          

This module provides a class `color_normalization` to standardize WSIs using four color normalization methods:
    • Macenko
    • Vahadane
    • Reinhard
    • Histogram Matching
    
---------------------------------------------------------------------------
Author: Tanviben.Patel (tanviben.patel@fda.hhs.gov) SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
Version ='1.0'
---------------------------------------------------------------------------
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


class color_normalization:
    """
    A class for performing various color normalization techniques on Whole Slide Images (WSIs),
    including Macenko, Vahadane, Reinhard, and histogram matching methods.
    
    Attributes
    ----------
    stain_matrix_target : np.ndarray or None
        The stain matrix computed from the target image for normalization.
    target_means : list or None
        Mean values of LAB channels for the target image (used in Reinhard).
    target_stds : list or None
        Standard deviations of LAB channels for the target image (used in Reinhard).
    """
    def __init__(self):
        """
        Initializes the ColorNormalization object with default target stain matrix
        and color statistics.
        """
        self.stain_matrix_target = None
        self.target_means = None
        self.target_stds = None


    @staticmethod
    def show_colors(C):
        """
        Displays stain color vectors as horizontal RGB bars.

        Parameters
        ----------
        C : ndarray
            Array of stain vectors.
        """
        n = C.shape[0]
        for i in range(n):
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i] / 255 if C[i].max() > 1.0 else C[i], linewidth=20)
        plt.axis('off')
        plt.axis([0, 1, -1, n])

    @staticmethod
    def show(image, now=True, fig_size=(10, 10)):
        """
        Displays a scaled version of an image.

        Parameters
        ----------
        image : ndarray
            Input image.
        now : bool
            Whether to immediately show the plot.
        fig_size : tuple
            Figure size for the plot.
        """
        image = image.astype(np.float32)
        m, M = image.min(), image.max()
        if fig_size is not None:
            plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
        plt.imshow((image - m) / (M - m) if m != M else image, cmap='gray')
        plt.axis('off')
        if now:
            plt.show()

    @staticmethod
    def lab_split(I):
        """
        Splits an RGB image into L, A, and B channels in LAB color space.

        Parameters
        ----------
        I : ndarray
            RGB image.

        Returns
        -------
        tuple
            Split LAB channels.
        """
        return cv.split(cv.cvtColor(I, cv.COLOR_RGB2LAB).astype(np.float32))

    @staticmethod
    def merge_back(I1, I2, I3):
        """
        Merges LAB channels back into an RGB image.

        Parameters
        ----------
        I1, I2, I3 : ndarray
            LAB channels.

        Returns
        -------
        ndarray
            Merged RGB image.
        """
        return cv.cvtColor(cv.merge((I1, I2, I3)).astype(np.uint8), cv.COLOR_LAB2RGB)

    @staticmethod
    def standardize_brightness(I):
        """
        Standardizes brightness of the image using 90th percentile normalization.

        Parameters
        ----------
        I : ndarray
            Input image.

        Returns
        -------
        ndarray
            Brightness-normalized image.
        """
        p = np.percentile(I, 90)
        return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

    @staticmethod
    def remove_zeros(I):
        """
        Replaces zero values in the image with one to prevent log(0) errors.

        Parameters
        ----------
        I : ndarray
            Input image.

        Returns
        -------
        ndarray
            Image with zeros replaced by ones.
        """
        return np.where(I == 0, 1, I)

    @staticmethod
    def RGB_to_OD(I):
        """
        Converts an RGB image to Optical Density (OD) space.

        Parameters
        ----------
        I : ndarray
            Input RGB image.

        Returns
        -------
        ndarray
            Image in OD space.
        """
        epsilon = 1e-6
        return -1 * np.log((color_normalization.remove_zeros(I) + epsilon) / 255)

    @staticmethod
    def OD_to_RGB(OD):
        """
        Converts an image from Optical Density (OD) space back to RGB.

        Parameters
        ----------
        OD : ndarray
            Image in OD space.

        Returns
        -------
        ndarray
            RGB image.
        """
        return (255 * np.exp(-1 * OD)).astype(np.uint8)

    @staticmethod
    def normalize_rows(A):
        """
        Normalizes the rows of a matrix to unit length.

        Parameters
        ----------
        A : ndarray
            Input matrix.

        Returns
        -------
        ndarray
            Row-normalized matrix.
        """
        return A / np.linalg.norm(A, axis=1, keepdims=True)

    @staticmethod
    def get_mean_std(I):
        """
        Computes mean and standard deviation of LAB channels.

        Parameters
        ----------
        I : ndarray
            RGB image.

        Returns
        -------
        tuple
            Means and standard deviations of LAB channels.
        """
        I1, I2, I3 = color_normalization.lab_split(I)
        return [np.mean(I1), np.mean(I2), np.mean(I3)], [np.std(I1), np.std(I2), np.std(I3)]

    def get_stain_matrix_M(self, I, beta=0.15, alpha=1):
        """
        Extracts stain matrix using Macenko's method.

        Parameters
        ----------
        I : ndarray
            Input RGB image.
        beta : float
            OD threshold for filtering.
        alpha : int
            Percentile cutoff for angle selection.

        Returns
        -------
        ndarray
            Stain matrix.
        """
        OD = self.RGB_to_OD(I).reshape((-1, 3))
        OD = OD[(OD > beta).any(axis=1), :]
        if OD.shape[0] == 0:
            raise ValueError("No valid OD values found in the image.")
        
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
        V = V[:, ::-1]
        V[:, 0] *= np.sign(V[0, 0])
        V[:, 1] *= np.sign(V[0, 1])

        That = np.dot(OD, V)
        minPhi, maxPhi = np.percentile(np.arctan2(That[:, 1], That[:, 0]), [alpha, 100 - alpha])
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi), 0]).T)
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi), 0]).T)

        dictionary = self.normalize_rows(np.array([v1, v2]))
        v3 = np.cross(dictionary[0], dictionary[1])
        return np.vstack([dictionary, v3])

    def get_stain_matrix_V(self, I):
        """
        Extracts stain matrix using Vahadane's method via SVD.

        Parameters
        ----------
        I : ndarray
            Input RGB image.

        Returns
        -------
        ndarray
            Stain matrix.
        """
        OD = self.RGB_to_OD(I).reshape((-1, 3))
        if OD.shape[0] < 3:
            raise ValueError("Not enough stain separation in image.")
        
        U, _, Vt = np.linalg.svd(OD, full_matrices=False)
        dictionary = self.normalize_rows(Vt[:2, :])
        return np.vstack([dictionary, np.cross(dictionary[0], dictionary[1])])
    @staticmethod
    def load_image(image_path):
        """
        Load an image from disk and convert it to RGB.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        image : ndarray
            RGB image.
        """
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        else:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image

    @staticmethod
    def plot_images(source, target, normalized, source_title="Source", target_title="Target", normalized_title="Normalized"):
        """
        Plot source, target, and normalized images side by side.

        Parameters
        ----------
        source : ndarray
            Source image.
        target : ndarray or None
            Target image.
        normalized : ndarray
            Normalized output image.
        """
        fig, axes = plt.subplots(1, 3 if target is not None else 2, figsize=(18, 6))
        axes[0].imshow(source)
        axes[0].set_title(source_title)
        axes[0].axis("off")

        if target is not None:
            axes[1].imshow(target)
            axes[1].set_title(target_title)
            axes[1].axis("off")
            axes[2].imshow(normalized)
            axes[2].set_title(normalized_title)
            axes[2].axis("off")
        else:
            axes[1].imshow(normalized)
            axes[1].set_title(normalized_title)
            axes[1].axis("off")

        plt.show()

    @staticmethod
    def plot_all_normalization_comparison(original_stats, normalized_stats_list, methods, target_image_path=None):
        """
        Compare all normalization methods by plotting before/after hue and saturation scatter plots.

        Parameters
        ----------
        original_stats : np.ndarray
            Hue/saturation statistics of the original images (mean/std hue and saturation).
        normalized_stats_list : list of np.ndarray
            List of hue/saturation statistics for normalized image sets.
        methods : list of str
            Names of the normalization methods (e.g., ["Macenko", "Vahadane", "Reinhard", "Histogram"]).
        """
        # Optional target point overlay
        target_point = None
        if target_image_path and os.path.exists(target_image_path):
            try:
                target_img = cv.imread(target_image_path)
                target_img = cv.cvtColor(target_img, cv.COLOR_BGR2RGB)
                hsv = cv.cvtColor(target_img, cv.COLOR_RGB2HSV)
                hue = hsv[:, :, 0].astype(np.float32) / 255.0
                sat = hsv[:, :, 1].astype(np.float32) / 255.0
                target_point = (np.mean(hue), np.mean(sat))
                print(f"Target image loaded. Mean Hue/Sat: {target_point}")
            except Exception as e:
                print(f"Failed to load target image: {e}")
                
        plt.rcParams["figure.figsize"] = (20, 10)
        plt.rcParams["font.size"] = 10

        fig, axes = plt.subplots(2, len(methods) + 1)

        # Before Standardization: Mean Hue vs. Mean Saturation
        axes[0, 0].scatter(original_stats[:, 0], original_stats[:, 2], c='r', alpha=0.5, s=20)
        if target_point:
            axes[0, 0].scatter(*target_point, c='blue', s=100, marker='X', label='Target')
        axes[0, 0].set_xlim((0, 1))
        axes[0, 0].set_ylim((0, 1))
        axes[0, 0].set_xlabel('Mean Hue Intensity')
        axes[0, 0].set_ylabel('Mean Saturation Intensity')
        axes[0, 0].set_title('Before')

        # Before Standardization: STD Hue vs. STD Saturation
        axes[1, 0].scatter(original_stats[:, 1], original_stats[:, 3], c='g', alpha=0.5, s=20)
        if target_point:
            axes[0, 0].scatter(*target_point, c='blue', s=100, marker='X', label='Target')
        axes[1, 0].set_xlim((0, 0.3))
        axes[1, 0].set_ylim((0, 0.3))
        axes[1, 0].set_xlabel('STD Hue Intensity')
        axes[1, 0].set_ylabel('STD Saturation Intensity')
        axes[1, 0].set_title('Before')

        # After Standardization: Plots for each method
        for i, stats in enumerate(normalized_stats_list):
            # Mean Hue vs. Mean Saturation
            axes[0, i + 1].scatter(stats[:, 0], stats[:, 2], c='r', alpha=0.5, s=20)
            if target_point:
                axes[0, i + 1].scatter(*target_point, c='blue', s=100, marker='X', label='Target')
            axes[0, i + 1].set_xlim((0, 1))
            axes[0, i + 1].set_ylim((0, 1))
            axes[0, i + 1].set_xlabel('Mean Hue Intensity')
            axes[0, i + 1].set_ylabel('Mean Saturation Intensity')
            axes[0, i + 1].set_title(f'{methods[i]}')

            # STD Hue vs. STD Saturation
            axes[1, i + 1].scatter(stats[:, 1], stats[:, 3], c='g', alpha=0.5, s=20)
            if target_point:
                axes[0, i + 1].scatter(*target_point, c='blue', s=100, marker='X', label='Target')
            axes[1, i + 1].set_xlim((0, 0.3))
            axes[1, i + 1].set_ylim((0, 0.3))
            axes[1, i + 1].set_xlabel('STD Hue Intensity')
            axes[1, i + 1].set_ylabel('STD Saturation Intensity')
            axes[1, i + 1].set_title(f'{methods[i]}')

        plt.tight_layout()
        plt.show()

    def process_single_image(self, source_image_path, target_image_path, method="Macenko"):
        """
        Normalize a single image using the specified method.

        Parameters
        ----------
        source_image_path : str
            Path to the source image.
        target_image_path : str
            Path to the target image.
        method : str
            Normalization method: "Macenko", "Vahadane", "Reinhard", or "Histogram".

        Returns
        -------
        tuple
            (source image, target image, normalized image)
        """
        print(f"Processing source image: {source_image_path}")
        try:
            source_image = self.load_image(source_image_path)
            target_image = self.load_image(target_image_path) if target_image_path else None
        except ValueError as e:
            print(f"Skipping {source_image_path} due to an error: {e}")
            return None, None, None

        try:
            if method == "Macenko":
                self.fit_macenko(target_image)
                normalized_image = self.transform_macenko(source_image)
            elif method == "Vahadane":
                self.fit_vahadane(target_image)
                normalized_image = self.transform_vahadane(source_image)
            elif method == "Reinhard":
                self.fit_reinhard(target_image)
                normalized_image = self.transform_reinhard(source_image)
            elif method == "Histogram":
                if target_image is None:
                    print("Skipping histogram normalization: Target image required.")
                    return None, None, None
                normalized_image = self.transform_histogram(source_image, target_image)
            else:
                raise ValueError(f"Invalid normalization method: {method}")
        except Exception as e:
            print(f"Error applying {method} normalization to {source_image_path}: {e}")
            return None, None, None

        return source_image, target_image, normalized_image

    def process_image_folder(self, input_folder, output_base_folder, target_image_path):
        """
        Normalize all images in a folder using multiple normalization methods.

        Parameters
        ----------
        input_folder : str
            Folder containing input images.
        output_base_folder : str
            Folder where output images will be saved.
        target_image_path : str
            Path to the reference target image.
        """
        methods = ["Macenko", "Vahadane", "Reinhard", "Histogram"]

        method_dirs = {method: os.path.join(output_base_folder, method) for method in methods}
        for method, path in method_dirs.items():
            os.makedirs(path, exist_ok=True)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]

        if not image_files:
            raise ValueError("No image files found in the folder.")

        for i, image_file in enumerate(image_files):
            source_image_path = os.path.join(input_folder, image_file)

            for method in methods:
                source, target, normalized_image = self.process_single_image(source_image_path, target_image_path, method)

                if source is None or normalized_image is None:
                    continue

                output_path = os.path.join(method_dirs[method], image_file)
                cv.imwrite(output_path, cv.cvtColor(normalized_image, cv.COLOR_RGB2BGR))
                print(f"Saved {method} normalized image: {output_path}")

                if i < 4:
                    self.plot_images(source, target, normalized_image, "Source Image", "Target Image", f"{method} Normalized Image")

        print(f"\n Processed images from folder: {input_folder}")
        print(f" Normalized images saved to: {output_base_folder}")
    @staticmethod
    def compute_hue_saturation_stats(images):
        """
        Compute mean and standard deviation of Hue and Saturation channels.

        Parameters
        ----------
        images : list of ndarray
            List of RGB images.

        Returns
        -------
        stats : ndarray
            Nx4 array of [mean_hue, std_hue, mean_sat, std_sat].
        """
        stats = []
        for img in images:
            hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
            hue = hsv[:, :, 0].astype(np.float32) / 255.0
            sat = hsv[:, :, 1].astype(np.float32) / 255.0
            stats.append([np.mean(hue), np.std(hue), np.mean(sat), np.std(sat)])
        return np.array(stats) if stats else np.empty((0, 4))

    @staticmethod
    def load_images_from_folder(folder):
        """
        Load RGB images from a folder.

        Parameters
        ----------
        folder : str
            Directory containing image files.

        Returns
        -------
        images : list of ndarray
            List of loaded RGB images.
        filenames : list of str
            Corresponding filenames.
        """
        images, filenames = [], []
        if not os.path.exists(folder):
            print(f" Folder not found: {folder}")
            return images, filenames
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            img = cv.imread(path)
            if img is not None:
                images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                filenames.append(fname)
        return images, filenames

    @staticmethod
    def plot_scatter_grid(original_stats, normalized_stats, target_image_path=None):
        """
        Plot Hue/Saturation mean and std before/after normalization.

        Parameters
        ----------
        original_stats : ndarray
            Stats for original images.
        normalized_stats : ndarray
            Stats for normalized images.
        target_image_path : str, optional
            Path to the reference image for mean hue/saturation overlay.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import cv2 as cv

        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        labels = ['Mean Hue vs Sat', 'Std Hue vs Sat']
        limits = [(0, 1), (0, 0.3)]

        target_point = None
        if target_image_path and os.path.exists(target_image_path):
            try:
                target_img = cv.imread(target_image_path)
                target_img = cv.cvtColor(target_img, cv.COLOR_BGR2RGB)
                hsv = cv.cvtColor(target_img, cv.COLOR_RGB2HSV)
                hue = hsv[:, :, 0].astype(np.float32) / 255.0
                sat = hsv[:, :, 1].astype(np.float32) / 255.0
                target_point = (np.mean(hue), np.mean(sat))
                print(f"Target image loaded. Mean Hue/Sat: {target_point}")
            except Exception as e:
                print(f"Failed to load target image: {e}")

        for i in range(2):
            ax[0, i].scatter(original_stats[:, i], original_stats[:, i + 2], c='r', alpha=0.5, s=20, label='Original')
            if target_point and i == 0:
                ax[0, i].scatter(*target_point, c='blue', s=100, marker='X', label='Target')
                ax[0, i].legend()
            ax[0, i].set_xlim(limits[i])
            ax[0, i].set_ylim(limits[i])
            ax[0, i].set_xlabel(f'{"Mean" if i == 0 else "STD"} Hue')
            ax[0, i].set_ylabel(f'{"Mean" if i == 0 else "STD"} Saturation')
            ax[0, i].set_title(f'Before: {labels[i]}')

            ax[1, i].scatter(normalized_stats[:, i], normalized_stats[:, i + 2], c='g', alpha=0.5, s=20, label='Normalized')
            if target_point and i == 0:
                ax[1, i].scatter(*target_point, c='blue', s=100, marker='X', label='Target')
                ax[1, i].legend()
            ax[1, i].set_xlim(limits[i])
            ax[1, i].set_ylim(limits[i])
            ax[1, i].set_xlabel(f'{"Mean" if i == 0 else "STD"} Hue')
            ax[1, i].set_ylabel(f'{"Mean" if i == 0 else "STD"} Saturation')
            ax[1, i].set_title(f'After: {labels[i]}')

        plt.tight_layout()
        plt.show()
        
    def fit_macenko(self, target):
        """
        Fits the Macenko normalization model using a target image.

        Parameters
        ----------
        target : ndarray
            Target RGB image.
        """
        target = self.standardize_brightness(target)
        self.stain_matrix_target = self.get_stain_matrix_M(target)

    def transform_macenko(self, I):
        """
        Applies Macenko normalization to an input image.

        Parameters
        ----------
        I : ndarray
            Source RGB image.

        Returns
        -------
        ndarray
            Color-normalized image.
        """
        I = self.standardize_brightness(I)
        h, w, _ = I.shape
        stain_matrix_source = self.get_stain_matrix_M(I)
        OD_source = self.RGB_to_OD(I).reshape((-1, 3))
        source_concentrations = np.dot(OD_source, np.linalg.pinv(stain_matrix_source))
        transformed_OD = np.dot(source_concentrations, self.stain_matrix_target)
        transformed_OD = np.clip(transformed_OD, a_min=0, a_max=None)
        return self.OD_to_RGB(transformed_OD).reshape(h, w, 3)

    def fit_vahadane(self, target):
        """
        Fits the Vahadane normalization model using a target image.

        Parameters
        ----------
        target : ndarray
            Target RGB image.
        """
        if target is None:
            raise ValueError("Target image is required for Vahadane normalization.")
        target = self.standardize_brightness(target)
        self.stain_matrix_target = self.get_stain_matrix_V(target)

    def transform_vahadane(self, I):
        """
        Applies Vahadane normalization to an input image.

        Parameters
        ----------
        I : ndarray
            Source RGB image.

        Returns
        -------
        ndarray
            Color-normalized image.
        """
        I = self.standardize_brightness(I)
        h, w, _ = I.shape
        stain_matrix_source = self.get_stain_matrix_V(I)
        OD_source = self.RGB_to_OD(I).reshape((-1, 3))
        OD_source = np.clip(OD_source, 0, None)
        nmf = NMF(n_components=2, init='random', random_state=42)
        source_concentrations = np.dot(OD_source, np.linalg.pinv(stain_matrix_source))
        transformed_OD = np.dot(source_concentrations, self.stain_matrix_target)
        transformed_OD = np.clip(transformed_OD, a_min=0, a_max=None)
        return self.OD_to_RGB(transformed_OD).reshape(h, w, 3)

    def fit_reinhard(self, target):
        """
        Fits the Reinhard normalization model using a target image.

        Parameters
        ----------
        target : ndarray
            Target RGB image.
        """
        self.target_means, self.target_stds = self.get_mean_std(self.standardize_brightness(target))

    def transform_reinhard(self, I):
        """
        Applies Reinhard normalization to an input image.

        Parameters
        ----------
        I : ndarray
            Source RGB image.

        Returns
        -------
        ndarray
            Color-normalized image.
        """
        I1, I2, I3 = self.lab_split(self.standardize_brightness(I))
        means, stds = self.get_mean_std(I)
        norm_channels = [((ch - means[i]) * (self.target_stds[i] / stds[i])) + self.target_means[i] for i, ch in enumerate([I1, I2, I3])]
        norm_channels = [np.clip(ch, 0, 255) for ch in norm_channels]
        return self.merge_back(*norm_channels)

    def transform_histogram(self, I, reference_image):
        """
        Applies histogram matching to the input image to match the reference image.

        Parameters
        ----------
        I : ndarray
            Source RGB image.
        reference_image : ndarray
            Reference RGB image.

        Returns
        -------
        ndarray
            Histogram-matched image.
        """
        I_lab = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        ref_lab = cv.cvtColor(reference_image, cv.COLOR_RGB2LAB)

        I1, I2, I3 = cv.split(I_lab)
        ref1, ref2, ref3 = cv.split(ref_lab)

        I1_matched = self.match_histograms(I1, ref1)
        I2_matched = self.match_histograms(I2, ref2)
        I3_matched = self.match_histograms(I3, ref3)

        I_matched = cv.merge((I1_matched, I2_matched, I3_matched))
        return cv.cvtColor(I_matched, cv.COLOR_LAB2RGB)

    def match_histograms(self, source, reference):
        """
        Matches the histogram of a source channel to that of a reference channel.

        Parameters
        ----------
        source : ndarray
            Source channel.
        reference : ndarray
            Reference channel.

        Returns
        -------
        ndarray
            Histogram-matched channel.
        """
        source_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
        reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
        source_cdf = np.cumsum(source_hist).astype(np.float32)
        source_cdf /= source_cdf[-1]
    
        reference_cdf = np.cumsum(reference_hist).astype(np.float32)
        reference_cdf /= reference_cdf[-1]
    
        mapping = np.interp(source.flatten(), bins[:-1], np.interp(source_cdf, reference_cdf, bins[:-1]))
        matched_image = mapping.reshape(source.shape)
    
        return np.clip(matched_image, 0, 255).astype(np.uint8)

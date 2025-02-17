import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class WSI_ColorNormalization:
    def __init__(self):
        self.stain_matrix_target = None
        self.target_means = None
        self.target_stds = None

    @staticmethod
    def show_colors(C):
        n = C.shape[0]
        for i in range(n):
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i] / 255 if C[i].max() > 1.0 else C[i], linewidth=20)
        plt.axis('off')
        plt.axis([0, 1, -1, n])

    @staticmethod
    def show(image, now=True, fig_size=(10, 10)):
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
        return cv.split(cv.cvtColor(I, cv.COLOR_RGB2LAB).astype(np.float32))

    @staticmethod
    def merge_back(I1, I2, I3):
        return cv.cvtColor(cv.merge((I1, I2, I3)).astype(np.uint8), cv.COLOR_LAB2RGB)

    @staticmethod
    def standardize_brightness(I):
        p = np.percentile(I, 90)
        return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

    @staticmethod
    def remove_zeros(I):
        return np.where(I == 0, 1, I)

    @staticmethod
    def RGB_to_OD(I):
        return -1 * np.log(WSI_ColorNormalization.remove_zeros(I) / 255)

    @staticmethod
    def OD_to_RGB(OD):
        return (255 * np.exp(-1 * OD)).astype(np.uint8)

    @staticmethod
    def normalize_rows(A):
        return A / np.linalg.norm(A, axis=1, keepdims=True)

    @staticmethod
    def get_mean_std(I):
        I1, I2, I3 = WSI_ColorNormalization.lab_split(I)
        return [np.mean(I1), np.mean(I2), np.mean(I3)], [np.std(I1), np.std(I2), np.std(I3)]

    def get_stain_matrix_M(self, I, beta=0.15, alpha=1):
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
        OD = self.RGB_to_OD(I).reshape((-1, 3))
        if OD.shape[0] < 3:
            raise ValueError("Not enough stain separation in image.")
        U, _, Vt = np.linalg.svd(OD, full_matrices=False)
        dictionary = self.normalize_rows(Vt[:2, :])
        return np.vstack([dictionary, np.cross(dictionary[0], dictionary[1])])

    def fit_macenko(self, target):
        target = self.standardize_brightness(target)
        self.stain_matrix_target = self.get_stain_matrix_M(target)

    def transform_macenko(self, I):
        I = self.standardize_brightness(I)
        h, w, _ = I.shape
        stain_matrix_source = self.get_stain_matrix_M(I)
        OD_source = self.RGB_to_OD(I).reshape((-1, 3))
        source_concentrations = np.dot(OD_source, np.linalg.pinv(stain_matrix_source))
        transformed_OD = np.dot(source_concentrations, self.stain_matrix_target)
        transformed_OD = np.clip(transformed_OD, a_min=0, a_max=None)
        return self.OD_to_RGB(transformed_OD).reshape(h, w, 3)

    def fit_vahadane(self, target):
        self.stain_matrix_target = self.get_stain_matrix_V(self.standardize_brightness(target))

    def transform_vahadane(self, I):
        I = self.standardize_brightness(I)
        h, w, _ = I.shape
        stain_matrix_source = self.get_stain_matrix_V(I)
        transformed_OD = np.dot(np.dot(self.RGB_to_OD(I).reshape((-1, 3)), np.linalg.pinv(stain_matrix_source)), self.stain_matrix_target)
        return self.OD_to_RGB(transformed_OD).reshape(h, w, 3)

    def fit_reinhard(self, target):
        self.target_means, self.target_stds = self.get_mean_std(self.standardize_brightness(target))

    def transform_reinhard(self, I):
        I1, I2, I3 = self.lab_split(self.standardize_brightness(I))
        means, stds = self.get_mean_std(I)
        norm_channels = [((ch - means[i]) * (self.target_stds[i] / stds[i])) + self.target_means[i] for i, ch in enumerate([I1, I2, I3])]
        norm_channels = [np.clip(ch, 0, 255) for ch in norm_channels]
        return self.merge_back(*norm_channels)

    def transform_histogram(self, I):
        I_lab = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        I1, I2, I3 = cv.split(I_lab)
        I1 = cv.equalizeHist(I1.astype(np.uint8))
        I_normalized = cv.merge((I1, I2, I3))
        return cv.cvtColor(I_normalized, cv.COLOR_LAB2RGB)

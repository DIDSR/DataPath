import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Utility Functions
def show_colors(C):
    """
    Shows rows of C as colors (RGB)
    """
    n = C.shape[0]
    for i in range(n):
        plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i] / 255 if C[i].max() > 1.0 else C[i], linewidth=20)
    plt.axis('off')
    plt.axis([0, 1, -1, n])

def show(image, now=True, fig_size=(10, 10)):
    """
    Show an image (np.array) with proper scaling.
    """
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size is not None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.imshow((image - m) / (M - m) if m != M else image, cmap='gray')
    plt.axis('off')
    if now:
        plt.show()

def lab_split(I):
    return cv.split(cv.cvtColor(I, cv.COLOR_RGB2LAB).astype(np.float32))

def merge_back(I1, I2, I3):
    return cv.cvtColor(cv.merge((I1, I2, I3)).astype(np.uint8), cv.COLOR_LAB2RGB)

def standardize_brightness(I):
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

def remove_zeros(I):
    return np.where(I == 0, 1, I)

def RGB_to_OD(I):
    return -1 * np.log(remove_zeros(I) / 255)

def OD_to_RGB(OD):
    return (255 * np.exp(-1 * OD)).astype(np.uint8)

def normalize_rows(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)

def notwhite_mask(I, thresh=0.8):
    return (cv.cvtColor(I, cv.COLOR_RGB2LAB)[:, :, 0] / 255.0) < thresh

def get_mean_std(I):
    I1, I2, I3 = lab_split(I)
    return [np.mean(I1), np.mean(I2), np.mean(I3)], [np.std(I1), np.std(I2), np.std(I3)]

# Macenko Normalization
def get_stain_matrix_M(I, beta=0.15, alpha=1):
    """Extract stain matrix using Macenko’s method with correct Eigen Decomposition (EVD)."""
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[(OD > beta).any(axis=1), :]  # Remove background pixels

    if OD.shape[0] == 0:
        raise ValueError("No valid OD values found in the image.")

    # **Use Eigen Decomposition instead of SVD**
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, ::-1]  # Ensure correct order
    V[:, 0] *= np.sign(V[0, 0])
    V[:, 1] *= np.sign(V[0, 1])

    # **Macenko's Angular Percentile-Based Stain Selection**
    That = np.dot(OD, V)
    minPhi, maxPhi = np.percentile(np.arctan2(That[:, 1], That[:, 0]), [alpha, 100 - alpha])

    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi), 0]).T)
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi), 0]).T)

    # **Normalize Only v1 and v2 (Like Vahadane)**
    dictionary = normalize_rows(np.array([v1, v2]))

    # **Compute v3 as their cross-product, keeping it unnormalized**
    v3 = np.cross(dictionary[0], dictionary[1])

    # **Return correctly structured stain matrix**
    return np.vstack([dictionary, v3])

    """Extract stain matrix using Macenko’s method, following Vahadane's normalization approach. Use SVD instead of Eigen Decomposition """
    """OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[(OD > beta).any(axis=1), :]  # Remove background pixels

    if OD.shape[0] < 3:
        raise ValueError("Not enough stain separation in image.")

    # **Use SVD instead of Eigen Decomposition**
    U, _, Vt = np.linalg.svd(OD, full_matrices=False)

    # **Normalize only the first two stain vectors**
    dictionary = normalize_rows(Vt[:2, :])

    # **Compute the third vector as their cross-product (not normalized)**
    v3 = np.cross(dictionary[0], dictionary[1])

    # **Return stain matrix similar to Vahadane**
    return np.vstack([dictionary, v3])"""

# Vahadane Normalization
def get_stain_matrix_V(I, threshold=0.8):
    #mask = notwhite_mask(I, thresh=threshold).reshape((-1,))
    #OD = RGB_to_OD(I).reshape((-1, 3))[mask]
    OD = RGB_to_OD(I).reshape((-1, 3))
    if OD.shape[0] < 3:
        raise ValueError("Not enough stain separation in image.")
    U, _, Vt = np.linalg.svd(OD, full_matrices=False)
    dictionary = normalize_rows(Vt[:2, :])
    return np.vstack([dictionary, np.cross(dictionary[0], dictionary[1])])

# Normalization Classes
class MacenkoNormalizer:
    def __init__(self):
        self.stain_matrix_target = None

    def fit(self, target):
        # **Ensure brightness normalization**
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix_M(target)

    def transform(self, I):
        I = standardize_brightness(I)  # **Standardize brightness**
        h, w, c = I.shape
        
        stain_matrix_source = get_stain_matrix_M(I)

        OD_source = RGB_to_OD(I).reshape((-1, 3))
        source_concentrations = np.dot(OD_source, np.linalg.pinv(stain_matrix_source))

        # **Ensure proper mapping to the target stain matrix**
        transformed_OD = np.dot(source_concentrations, self.stain_matrix_target)

        transformed_OD = np.clip(transformed_OD, a_min=0, a_max=None)  # Ensure valid OD values
        transformed_RGB = OD_to_RGB(transformed_OD).reshape(h, w, 3)

        return transformed_RGB

class VahadaneNormalizer:
    def __init__(self):
        self.stain_matrix_target = None
    def fit(self, target):
        self.stain_matrix_target = get_stain_matrix_V(standardize_brightness(target))
    def transform(self, I):
        I = standardize_brightness(I)
        h, w, _ = I.shape
        stain_matrix_source = get_stain_matrix_V(I)
        transformed_OD = np.dot(np.dot(RGB_to_OD(I).reshape((-1, 3)), np.linalg.pinv(stain_matrix_source)), self.stain_matrix_target)
        return OD_to_RGB(transformed_OD).reshape(h, w, 3)

class ReinhardNormalizer:
    def __init__(self):
        self.target_means = None
        self.target_stds = None
    def fit(self, target):
        self.target_means, self.target_stds = get_mean_std(standardize_brightness(target))
    def transform(self, I):
        I1, I2, I3 = lab_split(standardize_brightness(I))
        means, stds = get_mean_std(I)
        norm_channels = [((ch - means[i]) * (self.target_stds[i] / stds[i])) + self.target_means[i] for i, ch in enumerate([I1, I2, I3])]
        # Clip values to valid range
        norm_channels = [np.clip(ch, 0, 255) for ch in norm_channels]
        return merge_back(*norm_channels)

class HistogramNormalizer:
    def __init__(self):
        pass

    def fit(self, target):
        """Fit function for consistency, no fitting needed for histogram normalization."""
        pass

    def transform(self, I):
        """Apply histogram equalization to normalize the color distribution."""
        I_lab = cv.cvtColor(I, cv.COLOR_RGB2LAB)  # Convert to LAB color space
        I1, I2, I3 = cv.split(I_lab)

        # Apply histogram equalization only on the L channel (brightness)
        I1 = cv.equalizeHist(I1.astype(np.uint8))

        # Merge back LAB channels and convert to RGB
        I_normalized = cv.merge((I1, I2, I3))
        return cv.cvtColor(I_normalized, cv.COLOR_LAB2RGB)


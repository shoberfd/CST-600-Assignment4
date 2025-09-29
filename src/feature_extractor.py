import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin

class HOGTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for extracting HOG features.
    """
    def __init__(self, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
        # Store HOG parameters
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data,
        # so we just return self.
        return self

    def transform(self, X, y=None):
        """
        Applies HOG feature extraction to each image in the input array.
        """
        def get_hog(image):
            # The input X will be a 3D array of images (num_images, height, width)
            return hog(image,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       visualize=False,
                       feature_vector=True)
        
        # Apply the HOG function to each image in the dataset
        return np.array([get_hog(image) for image in X])
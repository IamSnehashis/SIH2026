from pathlib import Path
import numpy as np
import rasterio


def read_raster(path):
    """
    Read a satellite raster image.

    Returns:
        image: NumPy array with shape (bands, height, width)
        profile: Raster metadata
    """
    path = Path(path)

    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile

    return image, profile


def normalize_image(image):
    """
    Normalize image values to the range [0, 1].
    """
    image = image.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value == min_value:
        return np.zeros_like(image)

    return (image - min_value) / (max_value - min_value)


def save_raster(path, image, profile):
    """
    Save a NumPy image as a GeoTIFF.
    """
    path = Path(path)

    profile = profile.copy()
    profile.update(
        dtype=rasterio.float32,
        count=image.shape[0]
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image.astype(np.float32))


if __name__ == "__main__":
    print("SRM preprocessing module OK")
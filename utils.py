import numpy as np
from PIL import Image

def save_image_array(images, filename):
    """
    Save an array of images as a single PNG file.
    images: numpy array of shape (n_classes, n_samples, channels, height, width) or (n_samples, channels, height, width)
    """
    images = np.clip((images + 1) * 127.5, 0, 255).astype(np.uint8)  # Denormalize
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
    if images.shape[1] == 3:  # channels_first to channels_last
        images = np.transpose(images, (0, 2, 3, 1))
    n = images.shape[0]
    rows = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((rows * images.shape[1], rows * images.shape[2], 3), dtype=np.uint8)
    for i in range(n):
        row = i // rows
        col = i % rows
        grid[row*images.shape[1]:(row+1)*images.shape[1], col*images.shape[2]:(col+1)*images.shape[2], :] = images[i]
    Image.fromarray(grid).save(filename)
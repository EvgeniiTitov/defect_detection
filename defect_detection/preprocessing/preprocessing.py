import cv2
import numpy as np

def grayscale(image):
    """Converts image to grayscale. Returns the image converted"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def bilateral(image):
    """Effective at noise removal while preserving edges. However, its slower compared to other filters, so be careful"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    new_image = cv2.bilateralFilter(img,9,75,75)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
    return new_image

def mean(image, kernel_size=9):
    """Takes the average of all pixels under kernel area and replaces the central element with this average.
       Blurs image to remove noise. Smooths edges of the image."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Convert to HSV. OpenCV opens in BGR
    new_img = cv2.blur(img, (kernel_size, kernel_size))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    return new_img

def gaussian(image, kernel_size=9):
    """Similar to mean but it involves a weighted average of the surrounding pixels and it has sigma parameter.
       Gaussian filter blurs the edges but it does a better job of preserving them compared to the mean filter"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    new_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    return new_img

def median(image, kernel_size=9):
    """Calculates the median of the pixel intensities that surround the centre pixel in a NxN kernel. The median
    pixel gets replaced with the new value. Does a better job of removing salt-n-pepper noise compared to the mean and
    Gaussian filters. Preserves edges of an image but does not deal with speckle noise."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    new_img = cv2.medianBlur(img, kernel_size)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    return new_img

def custom_sharpening(image):
    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]])
    img = cv2.filter2D(image, -1, kernel)
    return img

def metadata_cutoff(image_path):
    """
    image_path - path to an image
    Check if an image has any metadata associated with it. If true, check camera orientation, fix if required.
    """
    from PIL import Image
    from PIL.ExifTags import TAGS
    import os

    save_path, file_name = os.path.split(image_path)

    with Image.open(image_path) as image:
        if image._getexif():
            exif = dict((TAGS[k], v) for k, v in image._getexif().items() if k in TAGS)
            try:
                if exif["Orientation"] == 6:
                    image = image.rotate(-90, expand = True)
                    # Option 1: save image with the metadata changed
                    image.save()
            except:
                pass
            
        else:
            print("No metadata") 
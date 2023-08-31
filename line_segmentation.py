import skimage
import skimage.measure as skime
import skimage.morphology as skim

from collections import OrderedDict

import scipy.ndimage as ndi
import scipy.ndimage.morphology as morph

import numpy as np
import pandas as pd
import cv2

from scipy.signal import find_peaks

import matplotlib.pyplot as plt

def profile_based_area_segmentation(input_image, threshold_factor = 0.7, border = 5,
                                    ):
  image_data = np.array(input_image)
  
  x_profile = np.sum(image_data,axis=0)
  x_profile = x_profile/np.max(x_profile)

  y_profile = np.sum(image_data,axis=1)
  y_profile = y_profile/np.max(y_profile)

  # plt.plot(x_profile)
  # plt.plot(y_profile)
  # plt.show()

  x_peaks = find_peaks(x_profile,height= np.max(x_profile)*threshold_factor)
  y_peaks = find_peaks(y_profile,height= np.max(y_profile)*threshold_factor)

  x_border = x_peaks[0]
  xmin,xmax = x_border[0], x_border[-1]

  y_border = y_peaks[0]
  ymin,ymax = y_border[0], y_border[-1]
  
  cropped_image = input_image[ymin-border:ymax+border,xmin-border:xmax+border]
  
  return cropped_image


def background_segmentation(cropped_image, filtered_image):
  filtered_mask = np.zeros_like(filtered_image)
  filtered_mask[filtered_image>0] = 1
  dilated_mask = skim.binary_dilation(filtered_mask.astype(bool))

  mask_mean = np.mean(cropped_image[dilated_mask==True])

  mask = np.zeros_like(dilated_mask)
  mask[cropped_image>mask_mean] = True

  inv_mask = np.logical_not(mask)
  
  labeled_image, num_features = ndi.label(inv_mask)
  print()
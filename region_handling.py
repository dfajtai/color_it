import os,sys,glob,re
import itertools
import skimage
import skimage.measure as skime
import skimage.morphology as skim

from collections import OrderedDict

import scipy.ndimage as ndi
import scipy.ndimage.morphology as morph

import numpy as np
import pandas as pd

import cv2
from PIL import ImageColor

from cropping import bbox_2d


def number_bbox_by_label(labeled_image, number_mask, tmp_dir = None, label_min_size = 250,
                         label_crop_contour = 5, number_fraction_limit = 0.5,
                         number_search_contour = 2,
                         save_tmp_imgs = False, number_border = [10,20]):
  
  if not isinstance(tmp_dir,type(None)):
    if not os.path.isdir(tmp_dir):
      os.makedirs(tmp_dir,exist_ok=True)

  assert isinstance(labeled_image,np.ndarray)
  assert isinstance(number_mask,np.ndarray)

  label_values = np.unique(labeled_image)

  label_info_dict_list = []

  final_labeled_image = np.zeros_like(labeled_image)

  for label in label_values:
    if label <= 1: # border label
      continue

    # if label == 16:
    #   print()
    # else:
    #   continue

    label_mask = labeled_image == label
    # crop mask

    x1,x2,y1,y2 = bbox_2d(label_mask, border = label_crop_contour)
    _label_mask = label_mask[x1:x2,y1:y2]
    if np.sum(_label_mask) < label_min_size:
      continue

    _number_mask = number_mask[x1:x2,y1:y2]
    
    contour_kernel = np.ones((3,3))
    dilated_label = morph.binary_dilation(_label_mask, structure=contour_kernel)
    label_outer_contour = np.logical_and(dilated_label, np.logical_not(_label_mask))
    label_inner_contour = np.logical_and(np.logical_not(morph.binary_erosion(_label_mask, structure=contour_kernel)),_label_mask)

    # fill holes resutlted by numbers.
    
    filled_label = ndi.binary_fill_holes(_label_mask,structure=contour_kernel)
    holes = np.logical_and(filled_label,np.logical_not(_label_mask))
    labeled_holes, hole_count = ndi.label(holes)
    
    background_holes = np.zeros_like(holes)
    for _i in range(hole_count): 
      hole_values = _i+1
      hole = labeled_holes == hole_values
      hole_number_count = np.sum((_number_mask&dilated_label)&hole)
      if hole_number_count >0 or np.sum(hole)<label_min_size:
        _label_mask[ndi.binary_dilation(hole)==1] = 1
      else:
        background_holes[ndi.binary_dilation(hole,structure=contour_kernel)==1] = 1
      
    # add loose contours
    new_number_mask = np.zeros_like(_number_mask)
    loose_contour_labels, loose_contour_count = ndi.label(np.logical_and(np.logical_or(label_inner_contour,label_outer_contour),np.logical_not(background_holes)))
    if loose_contour_count>1:
      new_number_mask[loose_contour_labels>1] = 1

    
    # update masks
    dilated_label = morph.binary_dilation(_label_mask, structure=contour_kernel)
    label_outer_contour = np.logical_and(dilated_label, np.logical_not(_label_mask))
    label_inner_contour = np.logical_and(np.logical_not(morph.binary_erosion(_label_mask, structure=contour_kernel)),_label_mask)
    
    # select numbers inside the label
    label_number_mask = _label_mask * _number_mask
    label_new_number_mask =  _label_mask * new_number_mask
    
    if np.sum(np.logical_and(label_outer_contour,_number_mask)) >= number_fraction_limit*np.sum(label_outer_contour):
      continue
    
    label_number_mask = np.max([label_number_mask,label_new_number_mask],axis = 0)

    dilated_label_number_mask = morph.binary_dilation(label_number_mask,structure=skim.disk(2),iterations=number_search_contour)
    labeled_numbers, number_label_count = ndi.label(dilated_label_number_mask)
    number_candidates = []
    for number_index in range(number_label_count):
      __number_mask = labeled_numbers == number_index+1
      number_bbox = np.array(bbox_2d(__number_mask, border = number_border)) + np.array([x1,x1,y1,y1])
      number_candidates.append(tuple(number_bbox))

    
    # save to tmp image
    if save_tmp_imgs:
      tmp_image = np.ones_like(_number_mask,dtype=np.uint8)
      rgb_tmp_img = cv2.cvtColor(tmp_image,cv2.COLOR_GRAY2RGB)
      rgb_tmp_img[:,:] = ImageColor.getrgb("black") 
      rgb_tmp_img[_label_mask==1] = ImageColor.getrgb("palegreen")
      rgb_tmp_img[label_outer_contour == 1] = ImageColor.getrgb("gray")
      rgb_tmp_img[label_inner_contour == 1] = ImageColor.getrgb("blue")
      rgb_tmp_img[dilated_label_number_mask==1] = ImageColor.getrgb("magenta")
      rgb_tmp_img[_number_mask == 1] = ImageColor.getrgb("white")
      rgb_tmp_img[label_number_mask == 1] = ImageColor.getrgb("lime")
      
      cv2.imwrite(os.path.join(tmp_dir,f"label_{str(int(label)).zfill(8)}.jpeg"),rgb_tmp_img)
    
    label_indices = np.where(_label_mask == 1)
    center = np.mean(label_indices,axis = 1)
    center += np.array([x1,y1])
    label_info_dict = OrderedDict(label = label, point_count = int(np.sum(_label_mask)), center = center.astype(int),
                                  number_candidates = number_candidates)
    
    label_info_dict_list.append(label_info_dict)
    final_labeled_image[x1:x2,y1:y2][_label_mask==1] = label

  
  label_info_df = pd.DataFrame(label_info_dict_list)
  label_info_df.to_csv(os.path.join(tmp_dir,"label_info.csv"),index=False)

  return final_labeled_image, label_info_df
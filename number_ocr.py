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

import pytesseract as pytes

from cropping import bbox_2d

import string


def ocr_image(image, fast = True, debug = False):
  if fast:
    candidate_string = pytes.image_to_string(image, 
                                             config=f'--psm 6 -c tessedit_char_whitelist={string.digits} -c tessedit_char_blacklist={string.ascii_letters}?\\/|' )
  else:
    candidate_string = pytes.image_to_string(image, 
                                             config=f'--tessdata-dir {os.getcwd()} --psm 6 --oem 1 -c tessedit_char_whitelist={string.digits} -c tessedit_char_blacklist={string.ascii_letters}?\\/|' )
  fixed_candidate_string = candidate_string.replace("\n\x0c","")
  
  string_correction_dict = {"A":"4","Z":"2","\\":"1","|":"1","[":"1","]":"1","g":"9","G":"6","%":"4","b":"6","2z":"2","f":"2","2)":"20","2U":"20"}
  for _char,_num in string_correction_dict.items():
    fixed_candidate_string = fixed_candidate_string.replace(_char,_num)
  if debug:
    print(fixed_candidate_string)
  candidate_number_match = re.match("\d+",fixed_candidate_string)
  if candidate_number_match:
    candidate_number = int(candidate_number_match.group(0))

    num_correction_dict = {0:6,65:6}
    if candidate_number in num_correction_dict.keys():
      candidate_number = num_correction_dict.get(candidate_number)

    return candidate_number



def perform_ocr_on_label_number_candidates(cropped_image, labeled_image, candidate_df, tmp_dir = None, save_tmp_imgs = False, 
                                           target_height = 200, debug = False):
  assert isinstance(cropped_image,np.ndarray)
  assert isinstance(labeled_image,np.ndarray)
  assert isinstance(candidate_df,pd.DataFrame)

  if not isinstance(tmp_dir,type(None)):
    if not os.path.isdir(tmp_dir):
      os.makedirs(tmp_dir,exist_ok=True)

  label_number_dict = {}
  possible_numbers = []

  for index, row in candidate_df.iterrows():
    if debug:
      print(row.to_dict())
    label = row.get("label")
    candidates = row.get("number_candidates")
    candidate_index = 1
    candidate_numbers = []

    if len(candidates) == 0:
      continue

    for c in candidates:
      if len(c)!=4:
        continue
      
      # crop number
      crop = cropped_image[c[0]:c[1],c[2]:c[3]]
      crop[ndi.binary_dilation(labeled_image[c[0]:c[1],c[2]:c[3]]!=label)] = 0

      # preprocess image
      threshold,binary = cv2.threshold(crop, 0, 255, cv2.THRESH_OTSU)

      binary = skim.remove_small_objects(binary>0,3,2)
      bbox = bbox_2d(binary, border = 0)
      crop = binary[bbox[0]:bbox[1],bbox[2]:bbox[3]].astype(np.uint8)

      # resize
      resize_factor = target_height / float(crop.shape[0])
      width = int(crop.shape[1] * resize_factor)
      height = int(crop.shape[0] * resize_factor)
      dim = (width, height)
      resized = cv2.resize(crop, dim, interpolation = cv2.INTER_CUBIC)
      resized = cv2.copyMakeBorder(resized,20,20,20,20,cv2.BORDER_CONSTANT,value=0)
      resized = cv2.medianBlur(resized,3)


      # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
      # resized = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel = kernel, iterations=1)
      # resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE,kernel = kernel,  iterations=1)

      # sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
      # resized = cv2.filter2D(resized, -1, sharpen_kernel)

      resized = 255 - (255 * resized)


      if save_tmp_imgs:
        tmp_img_path = os.path.join(tmp_dir,f"{str(int(label)).zfill(8)}_{candidate_index}.png")
        cv2.imwrite(tmp_img_path,resized)
      
      # perform ocr
      candidate_number = ocr_image(resized, debug = debug)
      if not isinstance(candidate_number,type(None)):
        candidate_numbers.append(candidate_number)

      candidate_index +=1
    
    for c in candidate_numbers:
      if c not in possible_numbers:
        possible_numbers.append(c)

    label_number_dict[label] = candidate_numbers


  return label_number_dict, possible_numbers
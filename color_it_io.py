import os,sys,glob,re


import cv2
import PIL
import numpy as np
import pandas as pd

from distinctipy import distinctipy

from pdf2image import convert_from_path

def convert_pdf_to_bitmap(input_pdf_path, dpi = 400, grayscale = True):
  out_image = None
  if not os.path.exists(input_pdf_path):
    return out_image
  image = convert_from_path(input_pdf_path, dpi = dpi)
  if not image:
    return image
  pil_image = image[0].convert('RGB') 
  open_cv_image = np.array(pil_image) 

  if not grayscale:
    return open_cv_image

  gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
  inverted = (255-gray)
  
  return inverted


def filter_image(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  sharpened_image = cv2.filter2D(image, -1, kernel)

  # downsampled = cv2.resize(sharpened_image,(0,0), fx = 0.5, fy=0.5)

  threshold,binary = cv2.threshold(sharpened_image,0,255,cv2.THRESH_OTSU)
  print(threshold)

  thinned = cv2.ximgproc.thinning(binary)

  return thinned


def create_graph_segmentation_image(sample_image, point_info_dict, branches):
  assert isinstance(point_info_dict,dict)
  grap_segment_image = np.zeros_like(sample_image)
  for branch in branches:
    for bp in branch:
      point_info = point_info_dict.get(bp)
      i,j = tuple(point_info.get("point_position"))
      grap_segment_image[i,j] = 255
  return grap_segment_image


def color_image_by_numbers(label_image, label_number_dict, possible_numbers, palette = None, pastel_factor = 0.8):
  assert isinstance(label_image,np.ndarray)
  assert isinstance(label_number_dict,dict)
  assert isinstance(possible_numbers,list)

  possible_numbers = sorted(possible_numbers)
  if not isinstance(palette, pd.DataFrame):
    number_of_colors = len(possible_numbers)
    colors = distinctipy.get_colors(number_of_colors, pastel_factor=pastel_factor)
    color_dict = {possible_numbers[i]:(np.array(colors[i])*255).astype(np.uint8).tolist() for i in range(number_of_colors)}
  else:
    palette["rgb"] = palette.apply(lambda x: (x.get("R"),x.get("G"),x.get("B")),axis = 1)
    palette_dict = {row.get("label"):row.get("rgb") for index,row in palette.iterrows() if row.get("missing")==False}
    color_dict = {}
    
    already_used_colors = []
    no_color_labels = []
    for p in possible_numbers:
      palette_color = palette_dict.get(p)
      if isinstance(palette_color,type(None)):
        no_color_labels.append(p)
      else:
        color_dict[p] = palette_color
        already_used_colors.append(tuple(np.array(palette_color)/255.0))
    
    generated_colors = distinctipy.get_colors(len(no_color_labels),exclude_colors=already_used_colors, pastel_factor=pastel_factor)
    for _i in range(len(no_color_labels)):
      _label = no_color_labels[_i]
      color_dict[_label] = tuple((np.array(generated_colors[_i])*255).astype(np.uint8))
    

  colored_image = cv2.cvtColor(np.ones_like(label_image,dtype=np.uint8)*255,cv2.COLOR_GRAY2RGB)
  for label, numbers in label_number_dict.items():
    if len(numbers) == 0:
      continue

    numbers = list(numbers)
    
    most_probable_number = max(set(numbers), key = numbers.count)

    label_mask = label_image == label

    if most_probable_number in color_dict.keys():
      colored_image[label_mask==1] = color_dict[most_probable_number]

  return colored_image

  
if __name__ == '__main__':
  pass
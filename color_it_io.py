import os,sys,glob,re


import cv2
import PIL
import numpy as np

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


if __name__ == '__main__':
  pass
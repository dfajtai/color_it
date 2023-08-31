import os,sys,glob,re
import pandas as pd
import json

from color_it_io import *
from line_segmentation import *
from graph_extraction import *
from find_numbers import *


def color_it_pipeline(input_pdf_list, out_dir):
  print(sampels)
  for s in input_pdf_list:
    print(s)
    input_img = convert_pdf_to_bitmap(s)
    # input_image_name = os.path.basename(s).replace(".pdf",".bmp")
    # cv2.imwrite(os.path.join(out_dir,input_image_name),input_img)

    cropped_img = profile_based_area_segmentation(input_img,border = 20)
    # cropped_image_name = os.path.basename(s).replace(".pdf","_cropped.bmp")
    # cv2.imwrite(os.path.join(out_dir,cropped_image_name),cropped_img)
    del(input_img)

    filtered_img = filter_image(cropped_img)
    filtered_image_name = os.path.basename(s).replace(".pdf","_filtered.jpeg")
    cv2.imwrite(os.path.join(out_dir,filtered_image_name),filtered_img)


    # labeled_background = background_segmentation(cropped_image=cropped_img,filtered_image=filtered_img)

    segment_numbers(input_path=s, filtered_img=filter_image,out_dir=out_dir)
    
    break



if __name__ == '__main__':
  sample_dir = "samples"
  sampels = glob.glob(os.path.join(sample_dir,"*.pdf"))

  color_it_pipeline(input_pdf_list=sampels, out_dir=sample_dir)
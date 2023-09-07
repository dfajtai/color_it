import os,sys,glob,re
import pandas as pd
import json

from color_it_io import *
from line_segmentation import *
from graph_extraction import *
from find_numbers import *
from region_handling import *
from number_ocr import *


def color_it_pipeline(input_pdf_list, out_dir):
  for s in input_pdf_list:
    print(s)
    iname = os.path.basename(s).replace(".pdf","")
    input_img = convert_pdf_to_bitmap(s)
    # input_image_name = iname+".jpeg"
    # cv2.imwrite(os.path.join(out_dir,input_image_name),input_img)

    cropped_img = profile_based_area_segmentation(input_img,border = 20)
    # cropped_image_name = iname + "_cropped.jpeg"
    # cv2.imwrite(os.path.join(out_dir,cropped_image_name),cropped_img)
    del(input_img)

    filtered_img = filter_image(cropped_img)
    filtered_image_name = iname+ "_filtered.jpeg"
    cv2.imwrite(os.path.join(out_dir,filtered_image_name),filtered_img)

    # background segmentation by labeling
    labeled_background = background_segmentation(cropped_image=cropped_img,filtered_image=filtered_img)

    # number segemntation by skeleton analysis
    number_mask = segment_numbers(input_path=s, filtered_img=filtered_img,out_dir=out_dir)

    # assign number cancidate bounding boxs to labels
    tmp_dir = os.path.join(out_dir,f"{iname}_labels")
    label_info_path = os.path.join(tmp_dir,"label_info.json")
    labeled_image_path = os.path.join(tmp_dir,"labeled.txt")
    if os.path.exists(label_info_path):
      label_info = pd.read_json(label_info_path)
      labeled_image = np.loadtxt(labeled_image_path,dtype=int)
    else:
      labeled_image, label_info = number_bbox_by_label(labeled_image=labeled_background, number_mask=number_mask,tmp_dir=tmp_dir,
                                                      save_tmp_imgs=True)
      label_info.to_json(label_info_path)
      np.savetxt(labeled_image_path, labeled_image, fmt='%d')


    # perform ocr on the cropped image using the bounding-box candidates
    tmp_dir = os.path.join(out_dir,f"{iname}_ocr")
    label_number_dict, possible_numbers = perform_ocr_on_label_number_candidates(cropped_image=cropped_img,labeled_image=labeled_image,
                                           candidate_df=label_info,tmp_dir=tmp_dir,
                                           save_tmp_imgs=True, debug = True)


    # color labels
    auto_colored_image = auto_color_image_by_numbers(label_image=labeled_image,label_number_dict=label_number_dict,possible_numbers=possible_numbers)
    auto_colored_image_path = os.path.join(out_dir,iname+"_auto_colored.png")
    cv2.imwrite(auto_colored_image_path,auto_colored_image)
  

if __name__ == '__main__':
  sample_dir = "samples"
  samples = glob.glob(os.path.join(sample_dir,"*.pdf"))
  print(samples)
  color_it_pipeline(input_pdf_list=samples, out_dir=sample_dir)
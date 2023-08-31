import os,sys,glob,re
import pandas as pd
import json

from color_it_io import *
from line_segmentation import *
from graph_extraction import *


def segment_numbers(input_path, filtered_img, out_dir):
  # examine centerline ( cardinality, neighbourhood, and so on)
  centerline_json_name= os.path.basename(input_path).replace(".pdf","_centerline.json")
  centerline_df_path = os.path.join(out_dir,centerline_json_name)
  if os.path.exists(centerline_df_path):
    centerline_df = pd.read_json(centerline_df_path)
  else:
    I = np.zeros_like(filtered_img,dtype=bool)
    I[filtered_img>0] = 1
    centerline_df = extract_centerline_info(skeleton_image_data=I)
    centerline_df.to_json(centerline_df_path,indent = 2)

  # handle centerline
  point_info_df, point_position_to_id_dict, point_info_dict = point_index_to_id(centerline_df)

  # pointy branches 1,2,3,45,6,7,9
  pointy_branches = extract_pointy_numbers(point_info_df=point_info_df,point_info_dict=point_info_dict)
  pointy_image = create_graph_segmentation_image(sample_image=filtered_img,
                          point_info_dict=point_info_dict,
                          branches= pointy_branches
                          )

  pointy_image_name = os.path.basename(input_path).replace(".pdf","_pointy.jpeg")
  cv2.imwrite(os.path.join(out_dir,pointy_image_name),pointy_image)

  # loopy branches 0,8,6,9,
  loop_branches = extract_looped_numbers(point_info_df=point_info_df,point_info_dict=point_info_dict)
  loopy_image = create_graph_segmentation_image(sample_image=filtered_img,
                          point_info_dict=point_info_dict,
                          branches= loop_branches
                          )
  loopy_image_name = os.path.basename(input_path).replace(".pdf","_loopy.jpeg")
  cv2.imwrite(os.path.join(out_dir,loopy_image_name),loopy_image)

  merged_image = np.max([pointy_image, loopy_image],axis = 0)
  merged_image_name = os.path.basename(input_path).replace(".pdf","_pointy_loopy.jpeg")
  cv2.imwrite(os.path.join(out_dir,merged_image_name),merged_image)

  number_mask = np.zeros_like(merged_image)
  number_mask[merged_image>0] = 1
  return number_mask


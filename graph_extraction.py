"""
Graph search algorythm to create a VessMorphoVis compatible representation of morphology based vessel skeleton

I REALLY hope, that this version will work...

nomenclature:
  endpoint : point with cardinality = 1
  midpoint : point with cardinality = 2
  crosspoint : point with cardinality > 2
  cross_endpoint : branch endpoint which is a cross_point
  crucial point: cross_endpoint or end_point

possible scenarios:
  E(/C) -> E(/C)
  E(/C) -> M -> [ .. -> M ] -> E(/C)
"""
import numpy as np

import skimage
import skimage.measure as skime
import skimage.morphology as skim

from collections import OrderedDict

import scipy.ndimage as ndi
import scipy.ndimage.morphology as morph

import pandas as pd

def extract_centerline_info(skeleton_image_data, remove_zero=True):
  """calculates local skeleton information for every skeleton points"""
  assert isinstance(skeleton_image_data, np.ndarray)


  cardinality_kernel = np.ones((3,3))
  cardinality_kernel[1,1] = 0
  # sum cardinality as ceratain point
  cardinality_matrix = ndi.convolve(skeleton_image_data.astype(np.uint8),cardinality_kernel,mode="constant",cval = 0)

  #neighbour matrices
  stacked_neighbour_matrix = np.zeros((2,int(np.power(3,2)-1),skeleton_image_data.shape[0],skeleton_image_data.shape[1]))
  
  index_matrix = np.indices(skeleton_image_data.shape) # index values
  index_displacement_matrix = np.array([np.ones(skeleton_image_data.shape),np.ones(skeleton_image_data.shape)]) # displacement matrix

  k = 0
  for i in range(3):
    for j in range(3):
      if i == 1 and j == 1:
        continue
      neighbour_kernel = np.zeros((3,3))
      neighbour_kernel[i,j] = 1
      index_displacement_matrix[0,:,:] = i-1
      index_displacement_matrix[1,:,:] = j-1

      neighbour_matrix = ndi.convolve(skeleton_image_data.astype(np.uint8),neighbour_kernel,mode = "constant",cval=0)
      stacked_neighbour_matrix[:,k,:,:] = neighbour_matrix.astype(np.int16)*(index_matrix-index_displacement_matrix)
      k+=1
  
  stacked_neighbour_matrix = stacked_neighbour_matrix.astype(np.int16)

  if remove_zero:
    points = np.array(np.argwhere((cardinality_matrix*skeleton_image_data)>0)) # remove points with no neighbours
  else:
    points = np.array(np.argwhere(skeleton_image_data == 1))

  point_info_df_list = []

  print("Extracting cardinality info...")

  for _i in np.arange(points.shape[0]):
    i, j = points[_i, :]
    neighbour_indices = [stacked_neighbour_matrix[:,k,i,j].tolist() for k in range(stacked_neighbour_matrix.shape[1]) if sum(stacked_neighbour_matrix[:,k,i,j])>0]
    point_info_df_list.append(OrderedDict(point_id= int(_i+1), point_position = points[_i, :], cardinality = int(cardinality_matrix[i,j]),
             neighbour_positions =  sorted(neighbour_indices))) 

    if _i % 100 == 0:
      print(f"{_i}/{points.shape[0]}") 

  point_info_df = pd.DataFrame(point_info_df_list)
  point_info_df = point_info_df.astype({'point_id': 'int32', "cardinality": "int32"})

  return point_info_df

def centerline_get_neighbours(df, point_index):
    N = sorted(df[df["point_id"] == point_index].neighbour_ids.tolist())
    if len(N) == 0:
        return []
    return N[0]

def point_index_to_id(point_info_df):
  assert isinstance(point_info_df,pd.DataFrame)
  point_position_to_id_dict = dict([(tuple(point_position),point_id) for point_position, point_id in zip(point_info_df["point_position"].tolist(),point_info_df["point_id"].tolist())])
  point_info_df["neighbour_ids"] = point_info_df["neighbour_positions"].apply(lambda x:[point_position_to_id_dict.get(tuple(_x)) for _x in x])
  point_info_dict = point_info_df.set_index("point_id", drop= True).to_dict(orient="index")
  return point_info_df, point_position_to_id_dict, point_info_dict


def extract_pointy_numbers(point_info_df,point_info_dict, max_neighbour_count = 3, max_length = 50):
  """
  Searches for 'pointy numbers' = numbers with endpoints: 1,2,3,4,5,6,7,9
  """
  assert isinstance(point_info_df,pd.DataFrame)
  assert isinstance(point_info_dict,dict)
  end_point_rows = point_info_df[point_info_df.cardinality == 1]
  end_points = sorted(end_point_rows["point_id"].tolist())
  endpoint_register = dict([(ep,False) for ep in end_points])

  branches = []
  touched_endpoints = 0
  
  while touched_endpoints<len(end_points):
    # search startingpoint from untouched endpoints
    starting_point = None
    i = touched_endpoints
    while i < len(end_points):
      if not endpoint_register.get(end_points[i]):
        starting_point = end_points[i]
        break
      i +=1
    
    if isinstance(starting_point,type(None)):
      print("No valid endpoint")
      break

    starting_point_info = point_info_dict.get(starting_point) 
    neighbours = starting_point_info.get("neighbour_ids")

    endpoint_register[starting_point] = True
    touched_endpoints+=1
    if len(neighbours) == 0:
      continue
    if len(neighbours) !=1:
      print("Endpoint assumption failed")
      continue

    pivot = neighbours[0]  # should have one neighbour by definition.
    branch = [starting_point]
    

    is_crossection = False
    pre_crossection_length = 1
    while not is_crossection:
      # depth-first search, until crosspoint reached.
      if isinstance(pivot,list):
        branch.extend(pivot)
        _neighbours = [set(point_info_dict.get(p).get("neighbour_ids")) for p in pivot ]
        neighbours = set.intersection(*_neighbours)
      else:
        branch.append(pivot)
        pivot_info = point_info_dict.get(pivot)
        neighbours = pivot_info.get("neighbour_ids")
      neighbours = [n for n in neighbours if n not in branch]
      is_crossection = len(neighbours)>max_neighbour_count
      pre_crossection_length+=1
      if len(neighbours) == 0:
        break
      pivot = neighbours[0] 

      if pre_crossection_length >max_length:
        break
    
    
    if len(neighbours)==0: # ended in an endpoint.
      endpoint_register[pivot] = True
      touched_endpoints+=1
    
    # drop long branches resulted by segmentation of the lines.
    if pre_crossection_length>max_length:
      continue

    print(branch)
    branches.append(branch)
  
  return branches

def extract_looped_numbers(point_info_df,point_info_dict, max_neighbour_count= 2, min_neighbour_count = 2, loop_iteration_count = 20 ):
  """
  Searches for 'looped numbers' = numbers with endpoints: 0,8,(6,9)
  """
  assert isinstance(point_info_df,pd.DataFrame)
  assert isinstance(point_info_dict,dict)
  crosspoint_rows = point_info_df[point_info_df.cardinality >= min_neighbour_count]
  crosspoints = sorted(crosspoint_rows["point_id"].tolist())
  crosspoint_register = dict([(rp,False) for rp in crosspoints])

  branches = []
  touched_crosspoints = 0
  
  while touched_crosspoints<len(crosspoints):
    # search startingpoint from untouched endpoints
    starting_point = None
    i = touched_crosspoints
    while i < len(crosspoints):
      if not crosspoint_register.get(crosspoints[i]):
        starting_point = crosspoints[i]
        break
      i +=1
    
    if isinstance(starting_point,type(None)):
      print("No valid endpoint")
      break

    starting_point_info = point_info_dict.get(starting_point) 
    neighbours = starting_point_info.get("neighbour_ids")

    crosspoint_register[starting_point] = True
    touched_crosspoints+=1
    if len(neighbours) == 0:
      continue

    pivot = neighbours  # should have one neighbour by definition.
    branch = [starting_point]
    

    is_crossection = False
    pre_crossection_length = 1
    search_iteration = 0

    loop_closed = False

    while search_iteration < loop_iteration_count and not is_crossection:
      # wide-first search in max of 'loop_iteration_count' steps
      if isinstance(pivot,list):
        branch.extend(pivot)
        _neighbours = [set(point_info_dict.get(p).get("neighbour_ids")) for p in pivot ]
        neighbours = list(set.union(*_neighbours))
      else:
        branch.append(pivot)
        pivot_info = point_info_dict.get(pivot)
        neighbours = pivot_info.get("neighbour_ids")
      neighbours = [n for n in neighbours if n not in branch]

      has_crossection = False
      for n in neighbours:
          if point_info_dict.get(n).get("cardinality") >= min_neighbour_count:
            touched_crosspoints+=1
            crosspoint_register[n] = True
            has_crossection = True

      # is_crossection = len(neighbours)>max_neighbour_count
      if not has_crossection:
        pre_crossection_length = len(branch)
      if len(neighbours) == 0: # loop is closed (no possible neighbours)
        loop_closed = True
        break  
      pivot = neighbours
      search_iteration+=1
    
    print(branch)
    if loop_closed: # only closed loops are valid.
      print(branch)
      branches.append(branch)
    elif pre_crossection_length>1:
      branches.append(branch[:pre_crossection_length])
  
  return branches
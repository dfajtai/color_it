import numpy as np

def bbox_2d(data,border = 0, min_border = 1):
  assert isinstance(data, np.ndarray)
  if not np.any(data):
      return 0,data.shape[0],0,data.shape[1]

  if not isinstance(border, list):
    border = np.ndim(data) * [border]
  else:
    assert len(border) == np.ndim(data)

  x = np.any(data, axis=(1,))
  y = np.any(data, axis=(0,))

  xmin, xmax = np.where(x)[0][[0, -1]]
  xmin, xmax = np.clip(np.array([xmin,xmax]) + np.array([-border[0]-min_border,+border[0]+min_border]),0,data.shape[0])[[0,1]]

  ymin, ymax = np.where(y)[0][[0, -1]]
  ymin, ymax = np.clip(np.array([ymin,ymax]) + np.array([-border[1]-min_border,+border[1]+min_border]),0,data.shape[1])[[0,1]]

  
  return xmin, xmax, ymin, ymax
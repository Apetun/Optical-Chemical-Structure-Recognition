import os
from collections import defaultdict
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

### globals 

THRESH_VAL = 100
LINE_WIDTH = 18 # needs to be even
BORDER = 30


STRUCTURES = [
    'struct19'
    ]


PATHS = ['data/' + structure + '/sd/' for structure in STRUCTURES]
TEMPLATES = ['train/oh/combined.png', 'train/or/combined.png', \
'train/o/combined.png', 'train/h/combined.png', 'train/n/combined.png', 'train/ro/combined.png']
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']
PICKLE_PATH = 'pickles/'

BOND_PATHS = ['train/single/', 'train/double/', 'train/triple/', 'train/dashed/', 'train/wedge/']
BOND_NAMES = ['single', 'double', 'triple', 'dashed', 'wedge']
COLOR_DICT = {
  'single':[255,0,0],
  'double':[0,0,255],
  'triple':[0,255,0],
  'dashed':[255,165,0],
  'wedge':[128,0,128],
  'none':[0,0,0]
}
COLOR_DICT_OCR = {
  'OH':[255,0,0],
  'OR':[0,255,0],
  'O':[0,0,255],
  'H':[255,255,0],
  'N':[0,255,255],
  'RO':[255,0,255]
}



def detect_bonds(img, template_dict, corner_file, bbox_width=40, angle_tol = 1):
  edges = []
  im = cv2.imread(img,0)
  # threshold the image to make binary
  ret,im = cv2.threshold(im,THRESH_VAL,255,cv2.THRESH_BINARY_INV)
  im = cv2.copyMakeBorder(im,BORDER,BORDER,BORDER,BORDER,cv2.BORDER_CONSTANT,0)
  with open(template_dict, 'rb') as handle:
    bbox_dict = pickle.load(handle)
  for k in bbox_dict.keys():
    for bbox in bbox_dict[k]:
      x0, x1, y0, y1 , _ = bbox
      im[y0:y1, x0:x1] = np.zeros((y1-y0,x1-x0))
  with open(corner_file, 'rb') as handle:
    corners = pickle.load(handle)
  corners = [tuple(corner[0]) if len(corner) > 0 else tuple(corner) for corner in corners]
  print(corners)
  display_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
  checked = set([])
  for corner1 in corners:
    dists = [np.linalg.norm(np.array(corner2)-np.array(corner1)) for corner2 in corners]
    dist_sort = np.argsort(dists)
    if len(dist_sort) < 4:
      max_idx = len(dist_sort)
    else:
      max_idx = 4
    for idx in dist_sort[0:max_idx]:
      corner2 = corners[idx]
      if corner1 == corner2:
        continue
      if (corner2,corner1) in checked:
        continue
      else:
        checked.add((corner1,corner2))
        v = np.array([corner2[0] - corner1[0], corner2[1] - corner1[1]])
        v_orth = np.array([corner1[1]-corner2[1], corner2[0] - corner1[0]])
        v_orth_norm = v_orth / np.linalg.norm(v_orth)
        corner1_vec = np.array(corner1)
        corner2_vec = np.array(corner2)
        p1 = corner1_vec + v_orth_norm*bbox_width*0.5
        p2 = corner1_vec - v_orth_norm*bbox_width*0.5
        p3 = corner2_vec - v_orth_norm*bbox_width*0.5
        p4 = corner2_vec + v_orth_norm*bbox_width*0.5
        mask = np.zeros(im.shape)
        point_list = np.array([p1,p2,p3,p4], dtype=np.int32)
        cv2.fillPoly(mask, [point_list], 1)
        flag = 0
        for corner in corners:
          if corner == corner1 or corner == corner2:
            continue
          if mask[int(corner[1]), int(corner[0])] != 0:
            flag = 1
      if flag == 1:
        continue
      line_detected = detect_bond_between_corners(im, corner1, corner2, bbox_width=bbox_width, angle_tol=angle_tol)
      if line_detected:
        edges.append((corner1, corner2))
        cv2.line(display_im, tuple(corner1), tuple(corner2), (0,0,255), 2)
  plt.imshow(display_im)
  plt.ion()
  plt.show()
  c = input("Correct? (y/n) --> ")
  n = input("Number of nodes --> ")
  if c == 'y':
    corr = 1.0
    fp_float = 0.0
    fn_float = 0.0
  else:
    corr = 0.0
    fp = input("False positives --> ")
    fn = input("False negatives --> ")
    fp_float = float(fp)
    fn_float = float(fn)
  plt.close()
  save_edge(edges,img)
  return corr, fp_float, fn_float, float(n)


# Save detected structures as pickle files
def save_edge(edges_list, image_name):
    pickle_filename = os.path.join(PICKLE_PATH, f"{image_name.split('/')[-1].split('.')[0]}_edges.pkl")
    with open(pickle_filename, 'wb') as f:
        pickle.dump(edges_list, f)

def detect_bond_between_corners(im, corner1, corner2, bbox_width, angle_tol, hough_tol=10, window_spacing=15):
  v = np.array([corner2[0] - corner1[0], corner2[1] - corner1[1]])
  v_norm = v / np.linalg.norm(v)
  v_orth = np.array([corner1[1]-corner2[1], corner2[0] - corner1[0]])
  v_orth_norm = v_orth / np.linalg.norm(v_orth)
  corner1_vec = np.array(corner1)
  corner2_vec = np.array(corner2)
  n_true = 0
  for degree in np.linspace(0,1,int(np.linalg.norm(v)/window_spacing),endpoint=False):
    new_im = im.copy()
    display_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    p1 = corner1_vec + degree*v + v_orth_norm*bbox_width*0.5
    p2 = corner1_vec + degree*v - v_orth_norm*bbox_width*0.5
    p3 = corner1_vec + degree*v - v_orth_norm*bbox_width*0.5 + 0.25*v
    p4 = corner1_vec + degree*v + v_orth_norm*bbox_width*0.5 + 0.25*v
    mask = np.zeros(im.shape)
    point_list = np.array([p1,p2,p3,p4], dtype=np.int32)
    cv2.fillPoly(mask, [point_list], 1)
    for y in range(im.shape[0]):
      for x in range(im.shape[1]):
        if mask[y,x] == 0:
          new_im[y,x] = 0
    for i,point in enumerate(point_list):
      point1 = point
      point2 = point_list[(i+1) % 4]
      cv2.line(display_im, tuple(point1), tuple(point2), color=[255,0,0], thickness=2)
  
    lines = cv2.HoughLines(new_im, 1, np.pi/180, hough_tol)
    line_detected = False
    try:
      original_theta = np.arctan((corner2[1]-corner1[1])/(corner2[0]-corner1[0])) + np.pi/2
    except ZeroDivisionError:
      original_theta = 0
    tol_radians = np.radians(angle_tol)
    if lines is not None:
      for rho,theta in lines[0]:
        if (abs(theta-original_theta) % np.pi) < tol_radians:
          line_detected = True
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 1000*(-b))
          y1 = int(y0 + 1000*(a))
          x2 = int(x0 - 1000*(-b))
          y2 = int(y0 - 1000*(a))
          cv2.line(display_im,(x1,y1),(x2,y2),(0,0,255),2)
    if line_detected:
      n_true += 1
    #plt.imshow(display_im)
    #plt.show()
  if n_true >= np.linalg.norm(v)/window_spacing-1:
    return True
  else:
    return False

for path in PATHS:
    corr_t, total, fp_t, fn_t, tp_t = 0.0, 0.0, 0.0, 0.0, 0.0
    for image in os.listdir(path):
        if image.lower().endswith('.png'):
            struct_name, img_num = image.split('_')[:2]  # Split to get structure and image number parts
            structure_file = f"{PICKLE_PATH}{struct_name}_{img_num.split('.')[0]}_structures.pkl"  # Construct pickle filename
            corner_file = f"{PICKLE_PATH}{struct_name}_{img_num.split('.')[0]}_corners.pkl"  # Construct pickle filename
            try:
                corr, fp, fn, tp = detect_bonds(os.path.join(path, image), structure_file,corner_file)
                corr_t += corr
                total += 1
                fp_t += fp
                fn_t += fn
                tp_t += tp
            except IOError:
                print(f"Failed to load pickle for {image}")
                continue
    print(f"Results for {path}: Correct - {corr_t}, Total - {total}, False Positives - {fp_t}, False Negatives - {fn_t}, True Positives - {tp_t}")
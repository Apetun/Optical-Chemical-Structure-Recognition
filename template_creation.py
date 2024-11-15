import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


LINE_WIDTH = 18
TEMPLATE_PATHS = [
'train/oh/',
'train/or/',
'train/o/',
'train/h/',
'train/n/',
'train/ro/'
]
SINGLE_TEMPLATE_PATHS = [
'train/o/',
'train/h/',
'train/n/',
'train/r/'
]
STRUCTURES = [
  'struct1', 
  'struct4', 
  'struct5',
  'struct8',
  'struct13',
  'struct16',
  'struct19',
  'struct20',
  'struct22',
]
PATHS = ['data/' + structure + '/sd/' for structure in STRUCTURES]
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']
SINGLE_TEMPLATE_NAMES = ['O', 'H', 'N', 'R']

TRAIN_IMAGES = ['01.png', '09.png', '17.png', '25.png', '33.png']
PYRAMID_SIZES = range(20,70,10)
STEP = 5

def get_max_size(x_list, y_list):
  s = max([max(x_list), max(y_list)])
  return (s,s)

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
def deskew(img, SZ):
  m = cv2.moments(img)
  if abs(m['mu02']) < 1e-2:
    return img.copy()
  skew = m['mu11']/m['mu02']
  M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
  img = cv2.warpAffine(img,M,(SZ, SZ))
  return img

def crop_and_resize_image(thresh_im, size_before_pad, pad):
  #plt.imshow(thresh_im, cmap='Greys_r')
  #plt.title("input")
  #plt.show()
  height, width = thresh_im.shape
  top_crop = 0
  bottom_crop = 0
  left_crop = 0
  right_crop = 0
  # top
  for y in range(height):
    row = thresh_im[y,:]
    if np.count_nonzero(row) > 0:
      top_crop = y
      break
  for y in reversed(range(height)):
    row = thresh_im[y,:]
    if np.count_nonzero(row) > 0:
      bottom_crop = y
      break
  for x in range(width):
    col = thresh_im[:,x]
    if np.count_nonzero(col) > 0:
      left_crop = x
      break
  for x in reversed(range(width)):
    col = thresh_im[:,x]
    if np.count_nonzero(col) > 0:
      right_crop = x
      break
  #print top_crop, bottom_crop, left_crop, right_crop
  fully_cropped = thresh_im[top_crop:bottom_crop, left_crop:right_crop]
  #plt.imshow(fully_cropped, cmap='Greys_r')
  #plt.title("fully cropped")
  #plt.show()
  fully_cropped = cv2.resize(fully_cropped, (size_before_pad, size_before_pad))
  fully_cropped = cv2.copyMakeBorder(fully_cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
  #plt.imshow(square,cmap='Greys_r')
  #plt.title("output")
  #plt.show()
  return fully_cropped

def crop_and_make_templates(path):
  ims = []
  for i, image in enumerate(os.listdir(path)):
    if image[len(image)-4:len(image)] != '.png':
      continue
    if 'combined' in image:
      continue
    full_name = path+image
    im = cv2.imread(full_name,0)
    ret,im = cv2.threshold(im,100,255,cv2.THRESH_BINARY_INV)
    im = crop_and_resize_image(im,40,0)
    ims.append(im)
  return ims

def stack_templates(path, train_split = 0.9):
  ims = crop_and_make_templates(path)
  n_images = float(len(ims))
  ims = [cv2.GaussianBlur(im,(5,5),5) / n_images for im in ims]
  final_image = ims[0]
  for im in ims[1:]:
    final_image = cv2.add(final_image,im)
  cv2.imwrite(path + 'combined.png', final_image)
  plt.imshow(final_image,cmap='Greys_r')
  plt.show()




for path in TEMPLATE_PATHS:
  stack_templates(path)


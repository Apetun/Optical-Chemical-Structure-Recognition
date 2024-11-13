from collections import defaultdict, Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



### globals 

THRESH_VAL = 100
LINE_WIDTH = 18 # needs to be even
BORDER = 30
STRUCTURES = [
  'struct19', 
]

PATHS = ['data/' + structure + '/sd/' for structure in STRUCTURES]
TEMPLATES = ['train/oh/combined.png', 'train/or/combined.png', \
'train/o/combined.png', 'train/h/combined.png', 'train/n/combined.png', 'train/ro/combined.png']
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']

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


## import training images into a dict

BOND_TRAINING_DICT = defaultdict(list)
for i,path in enumerate(BOND_PATHS):
  for image in os.listdir(path):
    if image[len(image)-4:len(image)] != '.png':
      continue
    BOND_TRAINING_DICT[BOND_NAMES[i]].append(path + image)

### ocr ground truth import ###

GROUND_TRUTH_DICT = {}
f = open('ocr_groundtruth.txt')
for line in f.readlines():
  split_line = line.split()
  k = split_line[0]
  vals = split_line[1:]
  vals = [int(v) for v in vals]
  GROUND_TRUTH_DICT[k] = vals
f.close()

### end ocr ground truth import ###

### corner ground truth import ###
CORNER_TRUTH_DICT = {}
g = open('corners_groundtruth.txt')
for line in g.readlines():
  split_line = line.split()
  k = split_line[0]
  v = split_line[1]
  CORNER_TRUTH_DICT[k] = int(v)
g.close()

## end corner ground truth import


### Bond classification

# Downscale images (0.7,0.7), then pad the images to 40 px width and clip edges
def preprocess_training(image_dict, size=(40,100), norm_width=40):
  processed = defaultdict(list)
  widths = defaultdict(list)
  avg_widths = defaultdict(list)
  avg_width_list = []
  for bond_type in image_dict.keys():
    imgs = image_dict[bond_type]
    for img in imgs:
      im = cv2.imread(img,0)
      widths[bond_type].append(im.shape[1])
  for key in widths:
    avg_width_list.append(np.mean(widths[key]))
    avg_widths[key] = np.mean(widths[key])
  max_width = max(avg_width_list)
  for bond_type in image_dict.keys():
    imgs = image_dict[bond_type]
    for img in imgs:
      im = cv2.imread(img,0)
      ret, im = cv2.threshold(im, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
      border = max(int((max_width-im.shape[1])/2),0)
      im = cv2.copyMakeBorder(im,0,0,border,border,cv2.BORDER_CONSTANT,0)
      im = cv2.resize(im,size)
      im = cv2.GaussianBlur(im,(5,5),5)
      #plt.imshow(im, cmap="Greys_r")
      #plt.show()
      center = im[20:80,:]
      processed[bond_type].append(center)
  return processed

# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
def hog(img):
  bin_n = 16
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
  mag, ang = cv2.cartToPolar(gx, gy)
  bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
  bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
  mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
  hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
  hist = np.hstack(hists)     # hist is a 64 bit vector
  return hist

def train_classifier(processed_dict, train_split = 0.9, type='svm'):
  label_conversion = defaultdict(str)
  label = 0
  
  featureX_train = []
  labels_train = []

  featureX_test = []
  labels_test = []

  for bond_type, im_list in processed_dict.items():
    label_conversion[label] = bond_type
    for im in im_list:
      if random.random() <= train_split:
        for i in range(0,60,10):
          features = hog(im[i:i+20,:])
          featureX_train.append(features)
          labels_train.append(label)
      else:
        for i in range(0,60,10):
          features = hog(im[i:i+20,:])
          featureX_test.append(features)
          labels_test.append(label)
    label += 1
  if type == 'svm':
    classifier = SVC(kernel='linear')
  if type == 'logistic_regression':
    classifier = LogisticRegression()
  if type == 'decision_tree':
    classifier = DecisionTreeClassifier()
  classifier.fit(featureX_train,labels_train)
  if train_split != 1:
    predicted = classifier.predict(featureX_test)
    hits_by_class = defaultdict(list)
    for i,label in enumerate(labels_test):
      if label == predicted[i]:
        hits_by_class[label].append(1)
      else:
        hits_by_class[label].append(0)
    for label, hits in hits_by_class.iteritems():
      print(label_conversion[label], np.mean(hits))
    return classifier.score(featureX_test, labels_test)
  return classifier, label_conversion

def get_bonds(line_segments, im):
  subimgs = []
  shape = im.shape
  for line_segment in line_segments:
    pt1 = line_segment[0]
    pt2 = line_segment[1]
    pt1y = int(pt1[0])
    pt1x = int(pt1[1])
    pt2y = int(pt2[0])
    pt2x = int(pt2[1])
    pt1vec = np.array([[pt1[0],pt1[1],1]]).transpose()
    pt2vec = np.array([[pt2[0],pt2[1],1]]).transpose()
    pt1vec2 = np.array([[pt1[0],pt1[1]]]).transpose()
    pt2vec2 = np.array([[pt2[0],pt2[1]]]).transpose()
    midpoint = (np.mean([pt1x,pt2x]), np.mean([pt1y,pt2y]))
    
    display = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(display,(pt1y-3,pt1x-3),(pt1y+3,pt1x+3),color=[255,0,0],thickness=-1)
    cv2.rectangle(display,(pt2y-3,pt2x-3),(pt2y+3,pt2x+3),color=[255,0,0],thickness=-1)
    #plt.imshow(display)
    #plt.show()

    translation = np.array([
      [1,0,-midpoint[1]+shape[1]/2],
      [0,1,-midpoint[0]+shape[0]/2]])

    pt1_t = np.dot(translation, pt1vec)
    pt2_t = np.dot(translation, pt2vec)
    # Convert the translated points to integers and scalars
    pt1y = int(pt1_t[0].item())
    pt1x = int(pt1_t[1].item())
    pt2y = int(pt2_t[0].item())
    pt2x = int(pt2_t[1].item())


    translated = cv2.warpAffine(im,translation,(shape[1], shape[0]))

    translated_display = cv2.cvtColor(translated, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(translated_display,(pt1y-3,pt1x-3),(pt1y+3,pt1x+3),color=[255,0,0],thickness=-1)
    cv2.rectangle(translated_display,(pt2y-3,pt2x-3),(pt2y+3,pt2x+3),color=[255,0,0],thickness=-1)
    #plt.imshow(translated_display)
    #plt.show()
    try:
      slope = float(pt1y-pt2y) / (pt1x-pt2x)
      angle = np.degrees(-np.arctan(slope))
    except ZeroDivisionError:
      angle = -90 
    
    dist = np.linalg.norm(pt2_t-pt1_t)
    rotated = ndimage.rotate(translated, angle, reshape=False)

    pt1y = max([int(shape[0]*0.5 - dist*0.5),0])
    pt2y = int(shape[0]*0.5 + dist*0.5)
    if pt2y < 0:
      pt2y = shape[1]
    pt1x = int(shape[1]*0.5 - 20)
    pt2x = int(shape[1]*0.5 + 20)

    #cv2.rectangle(rotated_display,(pt1x,pt1y),(pt2x,pt2y),color=[255,0,0],thickness=2)
    
    subimg = rotated[pt1y:pt2y, pt1x:pt2x]
    subimgs.append(subimg)
    #plt.imshow(subimg)
    #plt.show()
  return subimgs

def classify_bonds(edge_file,img,classifier,label_dict,template_dict_file,rect_w=6,empty_thresh=0.3):
  im = cv2.imread(img,0)
  ret,im = cv2.threshold(im,THRESH_VAL,255,cv2.THRESH_BINARY_INV)
  im = cv2.copyMakeBorder(im,BORDER,BORDER,BORDER,BORDER,cv2.BORDER_CONSTANT,0)
  shape = im.shape
  with open(edge_file,'rb') as handle:
    edges = pickle.load(handle)
  with open(template_dict_file,'rb') as handle:
    template_dict = pickle.load(handle)
  print(edges)
  print(template_dict)
  subimgs = get_bonds(edges, im)
  assignments = []
  for i,subimg in enumerate(subimgs):
    subimg = cv2.GaussianBlur(subimg, (5,5), 5)
    n_blocks = max(int(np.floor((subimg.shape[0]-10) / 10)), 0)
    blocks = []
    if n_blocks == 0:
      assignments.append('none')
      continue
    else:
      for block_start in [i*10 for i in range(n_blocks)]:
        block_end = block_start + 10
        blocks.append(hog(subimg[block_start:block_end, :]))
    guesses = classifier.predict(blocks)
    guess_count = Counter(guesses)
    label = (guess_count.most_common(1)[0])[0]
    print(label)
    assignments.append(label_dict[label])
  color_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
  for i,line_segment in enumerate(edges):
    color = COLOR_DICT[assignments[i]]
    if color == 'none':
      continue
    # Ensure points are integer tuples
    pt1 = (int(line_segment[0][0]), int(line_segment[0][1]))
    pt2 = (int(line_segment[1][0]), int(line_segment[1][1]))
    cv2.line(color_im,pt1,pt2,color,thickness=5)
  for key in template_dict.keys():
    if len(template_dict[key]) != 0:
      color = COLOR_DICT_OCR[key]
      for box in template_dict[key]:
        cv2.rectangle(color_im,(box[0],box[2]),(box[1],box[3]),color=color, thickness=5)
  plt.imshow(color_im)
  plt.ion()
  plt.show()
  c = input("Correct? (y/n) --> ")
  n = input("Number of edges correct --> ")
  if c == 'y':
    corr = 1.0
  else:
    corr = 0.0
  return corr, float(n) 

### Classifier testing ###
'''
scores = []
processed_dict = preprocess_training(BOND_TRAINING_DICT)
for i in range(10):
  scores.append(train_classifier(processed_dict, type = 'decision_tree'))
print np.mean(scores)
'''
###
    
processed_dict = preprocess_training(BOND_TRAINING_DICT)
classifier_svm, label_dict_svm = train_classifier(processed_dict, train_split=1)
classifier_decision, label_dict_decision = train_classifier(processed_dict, train_split=1)
classifier_logistic, label_dict_logistic = train_classifier(processed_dict, train_split=1)

for path in PATHS:
  corr_mol = 0.0
  corr_edge = 0.0
  total = 0.0
  for image in os.listdir(path):
    if image[len(image)-4:len(image)] != '.png':
      continue
 
    corr, n_corr = classify_bonds('pickles/' + image[0:11] + '_edges.pkl', path+image, classifier_svm, label_dict_svm, 'pickles/' + image[0:11] + '_structures.pkl')
    corr_mol += corr
    corr_edge += n_corr
    total += 1

  print(corr_mol, corr_edge, total)
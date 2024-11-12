import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

### globals 

THRESH_VAL = 100
LINE_WIDTH = 18  # needs to be even
BORDER = 30
STRUCTURES = [
    'struct19'
    ]

PATHS = ['data/' + structure + '/sd/' for structure in STRUCTURES]
PICKLE_PATH = 'pickles/'

def corner_detector(img, template_path, max_corners=20, display=True, rect_w=6):
    max_rgb_val = 255
    im = cv2.imread(img, 0)
    
    # Threshold the image to make binary
    _, im = cv2.threshold(im, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
    im = cv2.copyMakeBorder(im, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, 0)
    
    with open(template_path, 'rb') as handle:
        bbox_dict = pickle.load(handle)
    
    for k in bbox_dict.keys():
        for bbox in bbox_dict[k]:
            x0, x1, y0, y1 , _ = bbox
            im[y0:y1, x0:x1] = np.zeros((y1 - y0, x1 - x0))
    
    im = cv2.GaussianBlur(im, (LINE_WIDTH + 1, LINE_WIDTH + 1), LINE_WIDTH + 1)
    corners = cv2.goodFeaturesToTrack(im, max_corners, 0.0001, 35, blockSize=40, useHarrisDetector=True, k=0.04)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    
    if corners is None:
        return 0
    
    final_corners = []
    for corner in corners:
        corner_points = corner[0]
        corner_y, corner_x = int(corner_points[0]), int(corner_points[1])
        final_corners.append((corner_y, corner_x))
        cv2.rectangle(im, (corner_y - rect_w // 2, corner_x - rect_w // 2), 
                      (corner_y + rect_w // 2, corner_x + rect_w // 2), color=[255, 0, 0], thickness=-1)
    
    
    
    if display:
        plt.imshow(im)
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
            fp_float = float(input("False positives --> "))
            fn_float = float(input("False negatives --> "))
        plt.close()
        save_corner(corners, img)
        return corr, fp_float, fn_float, float(n)
    
    
# Save detected structures as pickle files
def save_corner(corner_list, image_name):
    print(image_name)
    pickle_filename = os.path.join(PICKLE_PATH, f"{image_name.split('/')[-1].split('.')[0]}_corners.pkl")
    with open(pickle_filename, 'wb') as f:
        pickle.dump(corner_list, f)



for path in PATHS:
    corr_t, total, fp_t, fn_t, tp_t = 0.0, 0.0, 0.0, 0.0, 0.0
    for image in os.listdir(path):
        if image.lower().endswith('.png'):
            struct_name, img_num = image.split('_')[:2]  # Split to get structure and image number parts
            pickle_file = f"{PICKLE_PATH}{struct_name}_{img_num.split('.')[0]}_structures.pkl"  # Construct pickle filename
            try:
                corr, fp, fn, tp = corner_detector(os.path.join(path, image), pickle_file)
                corr_t += corr
                total += 1
                fp_t += fp
                fn_t += fn
                tp_t += tp
            except IOError:
                print(f"Failed to load pickle for {image}")
                continue
    print(f"Results for {path}: Correct - {corr_t}, Total - {total}, False Positives - {fp_t}, False Negatives - {fn_t}, True Positives - {tp_t}")

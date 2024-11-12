import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# Constants
THRESH_VAL = 100  # Adjusted threshold value to reduce false positives
LINE_WIDTH = 18
BORDER = 30
STRUCTURES = ['struct19']
PATHS = [f'data/{structure}/sd/' for structure in STRUCTURES]
TEMPLATES = ['train/oh/combined.png', 'train/or/combined.png', 'train/o/combined.png', 'train/h/combined.png', 'train/n/combined.png', 'train/ro/combined.png']
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']
COLOR_DICT_OCR = {
    'OH': [255, 0, 0], 'OR': [0, 255, 0], 'O': [0, 0, 255],
    'H': [255, 255, 0], 'N': [0, 255, 255], 'RO': [255, 0, 255]
}

# Ensure the 'pickles' directory exists
if not os.path.exists('pickles'):
    os.makedirs('pickles')

### OCR ground truth import ###
GROUND_TRUTH_DICT = {}
with open('ocr_groundtruth.txt') as f:
    for line in f:
        split_line = line.split()
        k = split_line[0]
        vals = [int(v) for v in split_line[1:]]
        GROUND_TRUTH_DICT[k] = vals

### End OCR ground truth import ###

# Helper function to determine if a point is inside a box
def inside_box(center_x, center_y, box):
    return (center_x < box[1] and center_x > box[0] and center_y < box[3] and center_y > box[2])

# Template matching with scale variation and non-maximal suppression
def template_match(template, img, min_scale=0.3, max_scale=1.0, n_scales=15, threshold=0.6):
    im = cv2.imread(img, 0)
    ret, im = cv2.threshold(im, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
    im = cv2.copyMakeBorder(im, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, 0)
    im = cv2.GaussianBlur(im, (LINE_WIDTH // 2, LINE_WIDTH // 2), LINE_WIDTH // 2)
    tem = cv2.imread(template, 0)
    boxes = []

    # Scaling and matching at each scale level
    for scale in np.linspace(min_scale, max_scale, n_scales):
        tem_rescaled = cv2.resize(tem, None, fx=scale, fy=scale)
        w, h = tem_rescaled.shape[::-1]
        res = cv2.matchTemplate(im, tem_rescaled, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            score = res[pt[1], pt[0]]
            x0, x1, y0, y1 = pt[0], pt[0] + w, pt[1], pt[1] + h
            center_x, center_y = x0 + w // 2, y0 + h // 2
            flag, deletions = 0, []

            # Non-maximal suppression logic
            for i, box in enumerate(boxes):
                if inside_box(center_x, center_y, box) and box[4] > score:
                    flag = 1
                if inside_box(center_x, center_y, box) and box[4] < score:
                    deletions.append(i)
            if flag == 0:
                boxes.append((x0, x1, y0, y1, score))
            boxes = [boxes[i] for i in range(len(boxes)) if i not in deletions]
    return boxes

# Match templates across all scales for each template in the image
def all_template_match(templates, template_names, img, tol=0.6, display=False):
    template_dict, all_boxes, corresponding_templates = {}, [], []

    for i, template in enumerate(templates):
        boxes = template_match(template, img, threshold=tol)
        all_boxes += boxes
        corresponding_templates.extend([i] * len(boxes))

    # Apply non-maximal suppression to filter out overlapping boxes
    keep = [1 for _ in all_boxes]
    for i, box1 in enumerate(all_boxes):
        for j in range(i + 1, len(all_boxes)):
            box2 = all_boxes[j]
            center1x, center1y = (box1[0] + box1[1]) / 2, (box1[2] + box1[3]) / 2
            center2x, center2y = (box2[0] + box2[1]) / 2, (box2[2] + box2[3]) / 2
            if inside_box(center1x, center1y, box2) or inside_box(center2x, center2y, box1):
                if box1[4] >= box2[4]:
                    keep[j] = 0
                else:
                    keep[i] = 0

    for i, template in enumerate(templates):
        template_dict[template_names[i]] = [all_boxes[k] for k in range(len(all_boxes)) if corresponding_templates[k] == i and keep[k] == 1]

    # Optionally display the matched templates on the image
    acc = 0
    if display:
        im = cv2.imread(img, 0)
        ret, im = cv2.threshold(im, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
        im = cv2.copyMakeBorder(im, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, 0)
        im = cv2.GaussianBlur(im, (LINE_WIDTH // 2, LINE_WIDTH // 2), LINE_WIDTH // 2)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        for key, boxes in template_dict.items():
            color = COLOR_DICT_OCR.get(key, [0, 255, 0])
            for box in boxes:
                if box[4] > 0.8:  # Confidence threshold for display
                    cv2.rectangle(im, (box[0], box[2]), (box[1], box[3]), color=color, thickness=5)

        plt.imshow(im)
        plt.show()
        correct = input("Is this correct? (y/n)--> ")
        plt.close()
        if correct == 'y':
            acc = 1
    return template_dict, acc

# Save detected structures as pickle files
def save_structure(template_dict, image_name):
    pickle_filename = os.path.join('pickles', f"{image_name.split('.')[0]}_structures.pkl")
    print(template_dict)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(template_dict, f)

# Process all images and save results as pickles
def all_template_match_all_images(templates, template_names, path, tol=0.6, display=False):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    correct = 0
    n_images = 0
    for i, image in enumerate(os.listdir(path)):
        if not image.endswith('.png'):
            continue
        n_images += 1
        full_name = os.path.join(path, image)
        template_dict, acc = all_template_match(templates, template_names, full_name, tol=tol, display=display)
        
        # Save each detected structure as a pickle
        save_structure(template_dict, image)
        
        correct += acc
        comparison = [len(template_dict[key]) for key in TEMPLATE_NAMES]
        truth = GROUND_TRUTH_DICT.get(image[:8], [0] * len(TEMPLATE_NAMES))
        for i in range(len(comparison)):
            if comparison[i] == truth[i]:
                true_pos += comparison[i]
            elif comparison[i] > truth[i]:
                false_pos += comparison[i] - truth[i]
                true_pos += truth[i]
            elif comparison[i] < truth[i]:
                false_neg += truth[i] - comparison[i]
                true_pos += comparison[i]
    precision = float(true_pos) / (float(true_pos) + float(false_pos)) if true_pos + false_pos > 0 else 1.0
    recall = float(true_pos) / (float(true_pos) + float(false_neg)) if true_pos + false_neg > 0 else 1.0
    return precision, recall, true_pos, false_pos, false_neg, float(correct) / n_images

# Run the template matching and save results
for path in PATHS:
    precision, recall, tp, fp, fn, acc = all_template_match_all_images(
        TEMPLATES, TEMPLATE_NAMES, path, tol=0.77, display=True
    )
    print(precision, recall, acc)

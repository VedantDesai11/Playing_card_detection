import csv
import glob
from random import randint
import numpy as np
import cv2
from scipy import ndimage
from google.colab.patches import cv2_imshow
import math
import tqdm
import os


def place_image(img, bg, x, y):
    y1, y2 = y, y + img.shape[0]
    x1, x2 = x, x + img.shape[1]
    alpha_s = img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c])
    return bg


def load_card_resize_rotate(card_path, c_size, angle):
    img = cv2.imread(card_path, -1)
    img = cv2.resize(img, c_size)
    img = ndimage.rotate(img, angle)
    return img


def change_contrast(img, contrast):
    
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    return cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value < 0:
        lim = abs(value)
        v[v < lim] = 0
        v[v >= lim] -= abs(value)
    else:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def transform_point(position, point_x, point_y, angle, card_center_x, card_center_y, rotated_card_center_x, rotated_card_center_y):

    sth = math.sin(math.radians(abs(angle)))
    cth = math.cos(math.radians(abs(angle)))
    h1 = abs(card_center_y - point_y)
    h2 = abs(card_center_x - point_x)

    if angle > 0:
        if position == 0:
            a = rotated_card_center_x - h1*sth - h2*cth
            b = rotated_card_center_y - h1*cth + h2*sth
        if position == 1:
            a = rotated_card_center_x + h1*sth + h2*cth
            b = rotated_card_center_y + h1*cth - h2*sth
    else:
        if position == 0:
            a = rotated_card_center_x + h1*sth - h2*cth
            b = rotated_card_center_y - h1*cth - h2*sth
        if position == 1:
            a = rotated_card_center_x - h1*sth + h2*cth
            b = rotated_card_center_y + h1*cth + h2*sth

    return int(a), int(b)

from PIL import Image

kelvin_table = {
    3000: (255,180,107),
    4000: (255,209,163),
    5000: (255,228,206),
    6000: (255,243,239),
    7000: (245,243,255),
    8000: (227,233,255),
    9000: (214,225,255),
    10000: (204,219,255)
}

def convert_temperature(img, temp):
    image = Image.fromarray(img)
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    return np.array(image.convert('RGB', matrix))

def convert_to_yolo(xmax, ymax, xmin, ymin, img_width, img_height):
    w = (xmax - xmin) * (1/img_width)
    h = (ymax - ymin) * (1/img_height)
    x = ((xmax + xmin) / 2.0) * (1/img_width)
    y = ((ymax + ymin) / 2.0)* (1/img_height)

    return x, y, w, h

# set the size of the card and the final background image
card_height, card_width = 180, 120
bg_height, bg_width = 416, 416 # usually multiples of 32 as per YOLO object detection model
dataset_size = 52000

annotation_path = '/content/drive/MyDrive/Cards Dataset/labels_playing_card_detection.csv'

annotation_dict = {'info':['topleft_x', 'topleft_y', 'box_width', 'box_height', 'card_width', 'card_height']}

object_names = []
with open(annotation_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i,row in enumerate(reader):
        if i != 0:
            data = row[0].split(',')

            key = data[0]
            if key not in object_names:
                object_names.append(key)
            x, y, w, h = int(data[1]), int(data[2]), int(data[3]), int(data[4])
            card_w, card_h = int(data[6]), int(data[7])

            if key in annotation_dict:
                annotation_dict[key].append([x, y, w, h, card_w, card_h])
            else:
                annotation_dict[key] = [[x, y, w, h, card_w, card_h]]

save_dir = '/cards_dataset/'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)


with open(save_dir+'obj.names', 'w') as f:
    for c in object_names:
        if c == object_names[-1]:
            f.write("%s" % c)
        else: 
            f.write("%s\n" % c)

card_images_path = '/content/drive/MyDrive/Cards Dataset/Cards/*.png'
background_images_path = '/content/dtd/images/*/*.jpg'

cards = np.array(glob.glob(card_images_path))
bg_images = np.array(glob.glob(background_images_path))

idx = np.arange(0,52) 
b = np.arange(0,52) 
for i in range((dataset_size//52)-1):
    idx = np.concatenate((idx, b))

np.random.shuffle(idx)

for i in tqdm.tqdm(range(0, len(idx), 2)):

    bg_path = np.random.choice(bg_images, 1)[0]
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (bg_height, bg_width))
    label = []

    for j in range(2):
        card = cards[idx[i+j]]

        # get card name/labels from the path
        card_name = card.split("/")[-1].split(".")[0]
        card_rotation = randint(-90, 90)

        card_image = cv2.imread(card, -1)

        ratio_w, ratio_h = card_width/card_image.shape[1], card_height/card_image.shape[0]
        
        card_image = cv2.resize(card_image, (card_width, card_height))

        rotated_card = load_card_resize_rotate(card, (card_width, card_height), card_rotation)

        # card height and widths
        rotated_card_width, rotated_card_height = rotated_card.shape[1], rotated_card.shape[0]

        # Centers of rotated and normal card
        card_center_x, card_center_y = card_width//2, card_height//2
        rotated_center_x, rotated_center_y = rotated_card_width//2, rotated_card_height//2

        top_x = int(annotation_dict[card_name][0][0] * ratio_w)
        top_y = int(annotation_dict[card_name][0][1] * ratio_h)
        top_width = int(annotation_dict[card_name][0][2] * ratio_w)
        top_height = int(annotation_dict[card_name][0][3] * ratio_h)
        bottom_x = int(annotation_dict[card_name][1][0] * ratio_w)
        bottom_y = int(annotation_dict[card_name][1][1] * ratio_h)
        bottom_width = int(annotation_dict[card_name][1][2] * ratio_w)
        bottom_height = int(annotation_dict[card_name][1][3] * ratio_h)

        top_points = [[top_x, top_y], [top_x + top_width, top_y], [top_x, top_y + top_height], [top_x + top_width, top_y + top_height]]
        bottom_points = [[bottom_x, bottom_y], [bottom_x, bottom_y + bottom_height], [bottom_x + bottom_width, bottom_y], [bottom_x + bottom_width, bottom_y + bottom_height]]
        rotated_top_points = []
        rotated_bottom_points = []

        for k in range(4):
            x1, y1 = transform_point(0, top_points[k][0],top_points[k][1], card_rotation, card_center_x, card_center_y, rotated_center_x, rotated_center_y)
            rotated_top_points.append([x1, y1])

            x2, y2 = transform_point(1, bottom_points[k][0],bottom_points[k][1], card_rotation, card_center_x, card_center_y, rotated_center_x, rotated_center_y)
            rotated_bottom_points.append([x2, y2])

        limit_width = bg_width - rotated_card_width
        limit_height = bg_height - rotated_card_height

        if j == 0:
            offset_x = randint(0, bg_width//10)
            offset_y = randint(0, bg_height//10)
            
        else:
            offset_x = randint((bg_width//3), limit_width)
            offset_y = randint((bg_height//3), limit_height)
        
        top_max_x = max([i[0] for i in rotated_top_points]) + offset_x
        top_max_y = max([i[1] for i in rotated_top_points]) + offset_y
        top_min_x = min([i[0] for i in rotated_top_points]) + offset_x
        top_min_y = min([i[1] for i in rotated_top_points]) + offset_y
        bottom_max_x = max([i[0] for i in rotated_bottom_points]) + offset_x
        bottom_max_y = max([i[1] for i in rotated_bottom_points]) + offset_y
        bottom_min_x = min([i[0] for i in rotated_bottom_points]) + offset_x
        bottom_min_y = min([i[1] for i in rotated_bottom_points]) + offset_y

        bg = place_image(rotated_card, bg, offset_x, offset_y)
        # cv2.rectangle(bg, (top_min_x, top_min_y), (top_max_x, top_max_y), 255, 2)
        # cv2.rectangle(bg, (bottom_min_x, bottom_min_y), (bottom_max_x, bottom_max_y), 255, 2)
        
        index = object_names.index(card_name)
        p,q,r,s = convert_to_yolo(top_max_x, top_max_y, top_min_x, top_min_y, bg_width, bg_height)
        label.append(f'{index} {p} {q} {r} {s}')
        p,q,r,s = convert_to_yolo(bottom_max_x, bottom_max_y, bottom_min_x, bottom_min_y, bg_width, bg_height)
        label.append(f'{index} {p} {q} {r} {s}')

    num = str(i//2).zfill(7)
    img_name = f"/content/example_dataset/data/image_{num}.jpg"
    lbl_name = f"/content/example_dataset/data/image_{num}.txt"
    bg = change_brightness(bg, randint(-150,150))
    bg = change_contrast(bg, randint(-50,50))
    bg = convert_temperature(bg, (randint(3,10) * 1000))

    cv2.imwrite(img_name, bg)

    with open(lbl_name, 'w') as f:
        for line in label:
            if line == label[-1]:
                f.write("%s" % line)
            else: 
                f.write("%s\n" % line)
    

import glob
from random import randint
import numpy as np
import cv2
from scipy import ndimage
import csv


# Place img on bg image, given x offset and y offset
def place_image(img, bg, x, y):
	y1, y2 = y, y + img.shape[0]
	x1, x2 = x, x + img.shape[1]
	alpha_s = img[:, :, 3] / 255.0
	alpha_l = 1.0 - alpha_s

	for c in range(0, 3):
		bg[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c])

	return bg


def load_card_resize_rotate(card_path, c_size, angle):
	# read in RGBA, resize and rotate card
	img = cv2.imread(card_path, -1)
	img = cv2.resize(img, c_size)
	img = ndimage.rotate(img, angle)

	return img


def get_card_names(card_path):
	# get card name/labels from the path
	return card_path.split("/")[-1].split(".")[0]


def get_offsets(bg_width, bg_height, card_width, card_height):
	x, y = randint(0, (bg_width - card_width)), randint(0, (bg_height - card_height))
	return x, y


# # get path of all the card images
# cards = np.array(glob.glob('/Cards/*.png'))
#
# # select 2 cards at random from all cards
# picked_cards = np.random.choice(cards, 2, replace=False)
#
# # make random offsets for card 1 and card 2
# card_offsets = [[randint(0, (500 - 252)), randint(0, (500 - 252))], [randint(0, (500 - 252)), randint(0, (500 - 252))]]
#
# # make random rotations for card 1 and card 2
# card_rotations = [randint(-90, 90), randint(-90, 90)]

# define card size and bg size
card_size = (140, 210)
card_height = 210
card_width = 140
bg_height = 500
bg_width = 500
bg_size = (500, 500)

# # # GET ANNOTATIONS ----------
# annotation_path = 'labels_playing_card_detection.csv'
# annotation_dict = {}
#
# with open(annotation_path, newline='') as csvfile:
# 	reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
# 	for i, row in enumerate(reader):
# 		data = row[0].split(',')
# 		key = data[0]
# 		x, y, w, h = int(data[1]), int(data[2]), int(data[3]), int(data[4])
# 		card_w, card_h = int(data[6]), int(data[7])
# 		if key in annotation_dict:
# 			annotation_dict[key].append([x, y, w, h, card_w, card_h])
# 		else:
# 			annotation_dict[key] = [[x, y, w, h, card_w, card_h]]

card = './Cards/C9.png'

print(glob.glob('./Cards/*png'))

img = cv2.imread(card)
img = cv2.resize(img, card_size)

cv2.imshow('Card', img)
cv2.waitKey()

card_name = 'C2'

rotated_card = load_card_resize_rotate(card, card_size, -30)

ratio_w, ratio_h = card_width / annotation_dict[card_name][0][4], card_height / annotation_dict[card_name][0][5]

card_width, card_height = img.shape[1], img.shape[0]
rotated_card_width, rotated_card_height = rotated_card.shape[1], rotated_card.shape[0]
rotated_card_center_x, rotated_card_center_y = rotated_card.shape[1] // 2, rotated_card.shape[0] // 2

x1 = int(annotation_dict[card_name][0][0] * ratio_w)
y1 = int(annotation_dict[card_name][0][1] * ratio_h)
w1 = int(annotation_dict[card_name][0][2] * ratio_w)
h1 = int(annotation_dict[card_name][0][3] * ratio_h)
x2 = int(annotation_dict[card_name][1][0] * ratio_w)
y2 = int(annotation_dict[card_name][1][1] * ratio_h)
w2 = int(annotation_dict[card_name][1][2] * ratio_w)
h2 = int(annotation_dict[card_name][1][3] * ratio_h)

cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 1)
cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 1)

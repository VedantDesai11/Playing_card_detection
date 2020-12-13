# select 2 cards at random from all cards
picked_cards = np.random.choice(cards, 2, replace=False)

# get card name/labels from the path
card_names = [card.split("/")[-1].split(".")[0] for card in picked_cards]

# make random offsets for card 1 and card 2
card_offsets = [[randint(0, (500-252)) , randint(0, (500-252))],[randint(0, (500-252)) , randint(0, (500-252))]]

# make random rotations for card 1 and card 2
card_rotations = [randint(-90, 90), randint(-90, 90)]

# define card size and bg size
card_size = (140, 210)
bg_size = (500,500)

img = cv2.imread(picked_cards[0], -1)
img = cv2.resize(img, card_size)

card_1 = load_card(picked_cards[0], card_size, card_rotations[0])

card_name = card_names[0]
print(card_name, card_rotations[0])

w, h = 140/annotation_dict[card_name][0][4], 210/annotation_dict[card_name][0][5]

center_x, center_y = card_1.shape[1]//2, card_1.shape[0]//2

a1,b1,c1,d1 = int(annotation_dict[card_name][0][0]*w), int(annotation_dict[card_name][0][1]*h), int(annotation_dict[card_name][0][2]*w), int(annotation_dict[card_name][0][3]*h)
a2,b2,c2,d2 = int(annotation_dict[card_name][1][0]*w), int(annotation_dict[card_name][1][1]*h), int(annotation_dict[card_name][1][2]*w), int(annotation_dict[card_name][1][3]*h)

cv2.rectangle(img, (a1, b1), (a1+c1, b1+d1), (255,0,0), 1)
cv2.rectangle(img, (a2, b2), (a2+c2, b2+d2), (255,0,0), 1)
print(f"Top left box = ({a1},{b1}), Bottom right box = ({a2},{b2})")
print(f"Center of original card = {img.shape[1]//2}, {img.shape[0]//2}")
cv2_imshow(img)

h = img.shape[0]//2 - b1
a = h * sin()

cv2.circle(card_1, (card_1.shape[1]//2, card_1.shape[0]//2), 5, (255,0,0), 2)
cv2_imshow(card_1)
print(card_1.shape[0], card_1.shape[1])
print(a1,b1)
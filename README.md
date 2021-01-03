# playing_cards_dataset
 A synthetic dataset created for playing cards to be used to train object detection models.
 
 Cards folder contains cropped images of 52 cards as PNGs.
 labes.csv consists of annotations for these PNG cards for the top left, bottom right card number and suit.
 Format for labels :- C2 (card name), 21 (topleft corner of box, x), 19 (topleft corner of box, y), 37 (box width), 102 (box height), C2.png  (image name), 317 (card width), 489 (card height)
 
 create_dataset.py will make thr dataset by the specifed dataset size and according to the size of cards and final image. 

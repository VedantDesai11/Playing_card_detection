# playing_cards_dataset
 A synthetic dataset created for playing cards to be used to train object detection models.
 
 Cards folder contains cropped images of 52 cards as PNGs.  
 
 labes.csv consists of annotations for these PNG cards for the top left, bottom right card number and suit.  
 
 Format for labels :- 
   C2 (card name), 
   21 (topleft corner of box, x), 
   19 (topleft corner of box, y), 
   37 (box width), 102 (box height), 
   C2.png  (image name), 
   317 (card width), 
   489 (card height)
 
 create_dataset.py will make thr dataset by the specifed dataset size and according to the size of cards and final image. 
 
 YOLOv5_playing_cards.ipynb is the google colab code downloaded and can be used for training.  
 Blog used to implement YOLOv5 on custom dataset: https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/

## Training images created with labels look like this
### Image
![alt text](https://github.com/VedantDesai11/playing_cards_dataset/blob/main/image_0000001.jpg)
### Label
35 0.12740384615384617 0.21153846153846156 0.057692307692307696 0.09134615384615385
35 0.43149038461538464 0.41947115384615385 0.06009615384615385 0.0889423076923077
9 0.7199519230769231 0.5348557692307693 0.0889423076923077 0.09375
9 0.6850961538461539 0.8990384615384616 0.08653846153846154 0.09134615384615385

## Testing output looks like this
![alt text](https://github.com/VedantDesai11/playing_cards_dataset/blob/main/image_0000008.jpg)
![alt text](https://github.com/VedantDesai11/playing_cards_dataset/blob/main/image_0000027.jpg)

# Texture Dataset used for random backgrounds
M.Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, A. Vedaldi, "Describing Textures in the Wild"

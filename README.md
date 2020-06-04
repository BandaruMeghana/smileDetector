# smileDetector

The projeect aims at classifying a frame as either smiling or not smiling in real time. 

The model is trained on the SMILES dataset with each grayscale image of the dimension, 64 X 64 pixels. 

The challenges handled while modelling are
- The training dataset is tightly cropped around the face. But in real-time we'll have background as well. This is handling by first identifying the Region of Interest (RoI) and then classifiy. To get the RoI, I used the classic Haar Cascades Face detector. 
- The dataset is imbalanced. i.e., out of 13,165 images, we have 9475 not smiling images and only 3690 smiling images. To handle this data imbalance, I used the concept of balanced class weights while training. 

I created a LeNet architecture from scratch for training the model. Find this file in ```modules``` diredctory.  

```train_model.py``` uses the ```lenet.py``` for training and the ```smile_detector.py``` opens your webcam for real time classification

I'm able to get an accuracy of aroung __91%__ which can be further increased by tuning the hyper parameters. 



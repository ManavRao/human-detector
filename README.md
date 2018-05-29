##To run the model 
1. create the following directory structure
       
       human-detector
        |
        |_________ data
        |           |_______ models
        |           |_______ images
        |                       |_______ pos_person
        |                       |_______ neg_person
        |
        |__________ object_detector
        |
        |__________ output
       
2. clone the code in the object_detector directory
3. put your training dataset in the images directory in the corresponding folder
4. open terminal in object_detector directory
5. set and run the environment
6. run python extract_features.py
7. use the make file commands to train/run models


## Setting Environment
imutils

`sudo pip install imutils`

python-opencv

`sudo pip install opencv-python`

scikit-learn

`sudo pip install scikit-learn`

scikit-image

`sudo pip install scikit-image`



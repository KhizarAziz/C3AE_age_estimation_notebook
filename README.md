# C3AE: Age_estimation - Implemented in Notebook (Google Colab)
C3AE: Facial Age Estimation Project Implementation in Google Jupyer/Colab notebook.
A Very Easy Unofficial Implementation of C3AE(Age estimation) model in Google Colab using keras. I have kept the Code very simple & wrote too many comments, almost on every line.... for better understanding of beginners like me:) & whoever wants to build something on top it....


## Results

<div>
<img src="https://github.com/KhizarAziz/C3AE_keras/blob/master/result.jpg" width="200" height="200">
</div>



## Project Strcuture
```
C3AE-keras/
    |->datasets                (datasets should be placed here, e.g wiki_crop)
    |->detector                (contains face landmarks detector model. (used with dlib))
    |->model_saved             (will save weights here (best trained model weights)
    |->net_training            (contain scripts to define network and utils)
    |->preprocessing_scripts   (contain scripts for dataset preprocessing. initially added WIKI, IMDB & morph preprocessing modules. Which will load datasets, detect faces+landmarks, encode images & save as feather.)
```

## Code tested with
0. Google Colab
1. tensorflow : 2.3.0
2. keras : 2.4.3
3. cv2 : 4.1.2
4. dlib : 19.18.0
5. feather : 0.4.1
6. numpy : 1.18.5
7. pathlib


## Steps
Open "C3AE_Notebook_Implementation.ipynb" as a notebook, and start executing each cell step by step. as follows,

I have divided each part into sections. As follows.
1. Clone repo :  to get the code
2. Download Datasets :  This will Download datasets and extract into "C3ae_keras/datasets/".  (Initially contains links of wiki and imdb only, because morph is not publically available)
3. Preprocess datasets : Go to "preprocess_WIKI-IMDB.py" script first to change "dataset_name" varible to "wiki" or "imdb" for preprocessing coresponding dataset.
4. Train: 1. import libs 2. load pre processed dataset 3. train(0.8)/test(0.2)_split 4. Make Data Generator(because dataset is too large) 5. build net, train, and display history
5. Inference : Place an image in "/content/test.jpg" and run all cells in this section.


# Refrence and Credits
1. Original paper : [C3AE: Exploring the Limits of Compact Model for Age Estimation](https://arxiv.org/abs/1904.05059)
2. Starter code Taken from: StevenBanama's C3AE implementation (https://github.com/StevenBanama/C3AE). Great Work!

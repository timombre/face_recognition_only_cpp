# Face_regognition_only_cpp

Inspired by https://github.com/vinayakkailas/Face_Recognition in python
(for a working update, check my fork: https://github.com/timombre/Face_Recognition)

Face recognition using Tensorflow C++ API and OpenCV

#Requirements:

	1. Tensorflow built from sources and .so libraries generated (bazel build //tensorflow:libtensorflow_cc.so)
	2. OpenCV installed

Works on Unix systems (Not tested on Mac though)

## Using:

In Makefile, modify TF_BUILDDIR ?= with your Tensorflow directory.

make

Download Tensorflow weightfile and OpenCV Haarcascade:

wget https://github.com/arunmandal53/facematch/raw/master/20180408-102900/20180408-102900.pb

wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml



## Create a dataset of faces for each person 

sudo apt install ffmpeg 

ffmpeg -i /dev/video0 /PATH-FOR-VID/LABEL.mkv #modify dev/video0 if needed and choose your path

ffmpeg -i /PATH-FOR-VID/LABEL.mkv -vf fps=2 /PATH-FOR-IMAGES/LABEL_%04d.jpg -hide_banner

Arrangeyour dataset in below order

```
root folder  
│
└───Person 1
│   │───IMG1
│   │───IMG2
│   │   ....
└───Person 2
|   │───IMG1
|   │───IMG2
|   |   ....
```

cd prepdatabase

./prepdatabase.sh PATH_TO_YOUR_DATA_DIRECTORY

cd ..

./face_recognition face_embeddings_database.txt


## WIP

-The accuracy is very low, needs alignement implementation (in progress)




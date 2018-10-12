# Face_regognition_only_cpp

Inspired by https://github.com/vinayakkailas/Face_Recognition in python
(for a working update, check my fork: https://github.com/timombre/Face_Recognition)

Face recognition using Tensorflow C++ API and OpenCV

## Requirements :

	1. Tensorflow built from sources :
	Follow official guidelines
	https://www.tensorflow.org/install/install_sources
	
	Generate shared library: (bazel build //tensorflow:libtensorflow_cc.so)
	
	2. OpenCV and contribs installed:
	
	git clone https://github.com/opencv/opencv.git
	cd opencv 
	git checkout 3.3.1 
	cd ..
	git clone https://github.com/opencv/opencv_contrib.git
	cd opencv_contrib
	git checkout 3.3.1
	cd ..
	cd opencv
	mkdir build
	cd build
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_JPEG:BOOL=ON -DWITH_JPEG:BOOL=ON\
      -D BUILD_EXAMPLES=ON ..
	make
	sudo make install
	sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
	sudo ldconfig
	

Works on Unix systems (Not tested on Mac though)

## Using :

In Makefile, modify TF_BUILDDIR ?= with your Tensorflow directory.

    make

Download Tensorflow weightfile (The accuracy is better with older 20170512-110547.pb than with 20180408-102900.pb), Landmarks and OpenCV Haarcascade:

    wget https://gitlab.fit.cvut.cz/pitakma1/mvi-sp/raw/eb9c9db755077bd6fe0a61c1bbb1cced5f20d6d1/data/20170512-110547/20170512-110547.pb
    wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
    wget https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml


## Create a dataset of faces for each person :

    ./databasegenerator DATABASEFOLDER NAMEFORLABEL # you can tune the snapshot rate with the flag -fps "frequency", default is 1
    
Note : This is just for code testing, it is a very bad way to create a dataset as all pictures will have the same background and global features. If you already have independant pictures use them.

Arrange your dataset in below order :


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
## Create embeddings data base and run code :
   
    ./embed_database PATH_TO_YOUR_DATA_DIRECTORY 
    # you can add the -gen_aligned_db flag to generate the intermediary aligned database
    # you can also randomly (while conserving the global ratio within labels) split your dataset -splitdb "percentage" 
    # and -filesname to generate the list of the files ending in either dataset

    ./face_recognition face_embeddings_database.txt
    # you can add the -show_crop flag if you want to display the analyzed sub pictures
    # you can also tune the minimum display distance -thresh "float" (default is set at 0.4) and relative probability threshold -rp "float" (default is 1.25)
    # -mindist will give the original behaviour based on the minimum distance and not average and variance

    # In both cases, you can use the -choose_pb -choose_lm -choose_casc followed by the appropriate files to use your own pb, landmark and Haarcascade files.


## Stats :

With a split database, you can also somewhat evaluate your model

    ./stats face_embeddings_database.txt face_embeddings_database_testset.txt # -thresh "float"
    ./stats face_embeddings_database.txt -avg_db # will give you stats per label


## WIP :

-Rework stats for Linear Algebra enabled

-Average the result on several consecutive frames

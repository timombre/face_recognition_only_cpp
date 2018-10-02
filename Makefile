CC ?= cc
CXX ?= c++

TF_BUILDDIR ?= /home/tmenais/tensorflow

CODE_DIRS = prepdatabase/createembedding prepdatabase/listdatabase


CXXFLAGS := --std=c++17
INCLUDES := -I/usr/local/lib/python3.6/dist-packages/tensorflow/include/  -I$(TF_BUILDDIR) -I$(TF_BUILDDIR)/bazel-genfiles/ 
LIBS := -L$(TF_BUILDDIR)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework `pkg-config --libs --cflags opencv` -lstdc++fs -O3 -lm

all: face_recognition embed_database stats databasegenerator
	
face_recognition: src/libfacerecognition.cpp src/face_recognition.cpp src/statistics/libstats.cpp
	$(CXX) src/face_recognition.cpp src/libfacerecognition.cpp src/statistics/libstats.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o face_recognition

embed_database: src/libfacerecognition.cpp src/embed_database.cpp src/statistics/libstats.cpp
	$(CXX) src/embed_database.cpp src/libfacerecognition.cpp src/statistics/libstats.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o embed_database

stats: src/statistics/libstats.cpp src/statistics/stats.cpp
	$(CXX) src/statistics/stats.cpp src/statistics/libstats.cpp $(CXXFLAGS) -lstdc++fs -O3 -lm -o stats

databasegenerator: src/databasegeneration/databasegenerator.cpp
	$(CXX) src/databasegeneration/databasegenerator.cpp $(CXXFLAGS) `pkg-config --libs --cflags opencv` -lstdc++fs -O3 -lm -o databasegenerator

clean:
	rm -f face_recognition embed_database stats databasegenerator



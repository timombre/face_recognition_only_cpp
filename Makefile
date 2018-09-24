CC ?= cc
CXX ?= c++

TF_BUILDDIR ?= /home/tmenais/tensorflow

CODE_DIRS = prepdatabase/createembedding prepdatabase/listdatabase


CXXFLAGS := --std=c++17
INCLUDES := -I/usr/local/lib/python3.6/dist-packages/tensorflow/include/  -I$(TF_BUILDDIR) -I$(TF_BUILDDIR)/bazel-genfiles/ 
LIBS := -L$(TF_BUILDDIR)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework `pkg-config --libs --cflags opencv` -lstdc++fs -O3 -lm 
main: src/libfacerecognition.cpp src/listdatabase.cpp src/createembedding.cpp src/stats.cpp src/face_recognition.cpp
	$(CXX) src/listdatabase.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o listdatabase.out
	$(CXX) src/createembedding.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o createembedding.out
	$(CXX) src/stats.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o stats.out
	$(CXX) src/face_recognition.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o face_recognition

face_recognition: src/libfacerecognition.cpp src/face_recognition.cpp
	$(CXX) src/face_recognition.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o face_recognition

listdatabase: src/libfacerecognition.cpp src/listdatabase.cpp
	$(CXX) src/listdatabase.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o listdatabase.out

createembedding: src/libfacerecognition.cpp src/createembedding.cpp
	$(CXX) src/createembedding.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o createembedding.out

stats: src/libfacerecognition.cpp src/stats.cpp
	$(CXX) src/stats.cpp src/libfacerecognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o stats.out

clean:    

	rm -f face_recognition *.out



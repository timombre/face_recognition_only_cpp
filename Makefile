CC ?= cc
CXX ?= c++

TF_BUILDDIR ?= /home/tmenais/tensorflow

CODE_DIRS = prepdatabase/createembedding prepdatabase/listdatabase


CXXFLAGS := --std=c++17
INCLUDES := -I/usr/local/lib/python3.6/dist-packages/tensorflow/include/  -I$(TF_BUILDDIR) -I$(TF_BUILDDIR)/bazel-genfiles/ 
LIBS := -L$(TF_BUILDDIR)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework `pkg-config --libs --cflags opencv` -lstdc++fs -O3 -lm 
main: face_recognition.cpp
	
	TF_BUILDDIR=$(TF_BUILDDIR) $(MAKE) -C prepdatabase/createembedding;
	$(MAKE) -C prepdatabase/listdatabase;

	$(CXX) face_recognition.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o face_recognition

clean:

	
	$(MAKE) -C prepdatabase/createembedding clean ;
	$(MAKE) -C prepdatabase/listdatabase clean ;
    

	rm -f face_recognition



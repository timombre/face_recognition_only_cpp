CC ?= cc
CXX ?= c++

#TF_BUILDDIR := /home/tmenais/tensorflow

CXXFLAGS := --std=c++17
INCLUDES := -I/usr/local/lib/python3.6/dist-packages/tensorflow/include/  -I$(TF_BUILDDIR) -I$(TF_BUILDDIR)/bazel-genfiles/
LIBS := -L$(TF_BUILDDIR)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_videoio -lopencv_imgcodecs -lstdc++fs -O3 -lm 

main: createembedding.cpp
	$(CXX) createembedding.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o createembedding.out

clean:
	rm -f createembedding.out



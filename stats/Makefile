CC ?= cc
CXX ?= c++

CXXFLAGS := --std=c++17
LIBS := -lstdc++fs -O3 -lm

main: stats.cpp
	$(CXX) stats.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o stats.out

clean:
	rm -f stats.out
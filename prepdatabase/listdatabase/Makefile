CC ?= cc
CXX ?= c++

CXXFLAGS := --std=c++17
LIBS := -lstdc++fs -O3 -lm

main: listdatabase.cpp
	$(CXX) listdatabase.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o listdatabase.out

clean:
	rm -f listdatabase.out
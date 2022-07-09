
CXX = g++

CXXFLAGS = -O2 -std=c++11 -fPIC -Wall
CPPFLAGS = -I./include/
LIBS = -lstdc++

all: libtetris.so test

tetris.o: include/composer.h include/schedplan.h include/unique.h schedplan.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o tetris.o -c schedplan.cc

libtetris.so: tetris.o
	$(CXX) $(CPPFLAGS) -shared -o libtetris.so $^ ${LIBS}

test: test.cc libtetris.so
	$(CXX) $(CPPFLAGS) test.cc -o test -L./ -Wl,-rpath=./ -ltetris


clean:
	rm -r *.so
	rm -r *.o

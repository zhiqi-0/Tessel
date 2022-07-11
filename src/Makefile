
CXX = g++

CXXFLAGS = -O2 -std=c++11 -fPIC -Wall
CPPFLAGS = -I./include/
LIBS = -lstdc++

BUILD = build

all: ${BUILD}/libtetris.so test

${BUILD}/unique.o: include/unique.h unique.cc
	mkdir -p ${BUILD}
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o ${BUILD}/unique.o -c unique.cc

${BUILD}/schedplan.o: include/schedplan.h include/unique.h schedplan.cc
	mkdir -p ${BUILD}
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o ${BUILD}/schedplan.o -c schedplan.cc

${BUILD}/composer.o: include/unique.h include/schedplan.h include/composer.h composer.cc
	mkdir -p ${BUILD}
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o ${BUILD}/composer.o -c composer.cc

${BUILD}/libtetris.so: ${BUILD}/unique.o ${BUILD}/schedplan.o ${BUILD}/composer.o
	$(CXX) $(CPPFLAGS) -shared -o ${BUILD}/libtetris.so ${BUILD}/unique.o ${BUILD}/schedplan.o ${BUILD}/composer.o ${LIBS}

test: test.cc ${BUILD}/libtetris.so
	$(CXX) $(CPPFLAGS) test.cc -o test -L./build/ -Wl,-rpath=./build -ltetris


clean:
	rm -rf *.so
	rm -rf *.o
	rm test
	rm -rf ./build

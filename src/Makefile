
CXX = g++

CXXFLAGS = -O2 -std=c++11 -fPIC -Wall
CPPFLAGS = -I./include/
LIBS = -lstdc++

BUILD = build

all: ${BUILD}/libtetris.so ${BUILD}/test ${BUILD}/cases

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

${BUILD}/test: examples/test.cc ${BUILD}/libtetris.so
	$(CXX) $(CPPFLAGS) examples/test.cc -o ${BUILD}/test -L./build/ -Wl,-rpath=./build -ltetris


${BUILD}/cases: examples/cases.cc ${BUILD}/libtetris.so
	$(CXX) $(CPPFLAGS) examples/cases.cc -o ${BUILD}/cases -L./build/ -Wl,-rpath=./build -ltetris


clean:
	rm -rf ./build

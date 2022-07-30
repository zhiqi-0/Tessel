
CXX = g++

# -std=c++17 for g++ version >= 8
CXXFLAGS = -O2 -std=c++1z -fPIC -Wall -fopenmp
CPPFLAGS = -I./include/
LIBS = -lstdc++ -lpthread

BUILD = build

all: ${BUILD}/libtetris.so ${BUILD}/test ${BUILD}/cases

${BUILD}/unique.o: include/unique.h unique.cc
	mkdir -p ${BUILD}
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o ${BUILD}/unique.o -c unique.cc

${BUILD}/schedplan.o: include/schedplan.h include/unique.h schedplan.cc
	mkdir -p ${BUILD}
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o ${BUILD}/schedplan.o -c schedplan.cc

${BUILD}/composer.o: include/schedplan.h include/unique.h include/composer.h composer.cc
	mkdir -p ${BUILD}
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o ${BUILD}/composer.o -c composer.cc

${BUILD}/generalizer.o: include/schedplan.h include/unique.h include/composer.h include/generalizer.h generalizer.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o ${BUILD}/generalizer.o -c generalizer.cc

${BUILD}/libtetris.so: ${BUILD}/unique.o ${BUILD}/schedplan.o ${BUILD}/composer.o ${BUILD}/generalizer.o
	$(CXX) ${CXXFLAGS} $(CPPFLAGS) -shared -o ${BUILD}/libtetris.so ${BUILD}/unique.o ${BUILD}/schedplan.o ${BUILD}/composer.o ${BUILD}/generalizer.o ${LIBS}

${BUILD}/test: examples/test.cc ${BUILD}/libtetris.so
	$(CXX) $(CPPFLAGS) examples/test.cc -o ${BUILD}/test -L./build/ -Wl,-rpath=./build -ltetris


${BUILD}/cases: include/parser.h examples/cases.cc ${BUILD}/libtetris.so
	$(CXX) $(CPPFLAGS) examples/cases.cc -o ${BUILD}/cases -L./build/ -Wl,-rpath=./build -ltetris


clean:
	rm -rf ./build

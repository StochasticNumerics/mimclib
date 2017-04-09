TARGETS = _kuramoto.so

all: ${TARGETS}

_kuramoto.so: kuramoto.cpp
	g++ -O3 -fPIC -Wall -std=c++11 -shared -Wl,-soname,$@ -o $@ $^

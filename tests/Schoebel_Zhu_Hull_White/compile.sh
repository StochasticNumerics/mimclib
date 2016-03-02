#!/bin/bash

g++ -std=c++0x -c -fPIC ./szhw.cpp -o szhw.o
g++ -std=c++0x -shared -Wl,-soname,libszhw.so -o libszhw.so szhw.o

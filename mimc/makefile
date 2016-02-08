TARGETS =  _set_util.so libset_util.a
DEBUG = -O3
# DEBUG = -pg
# CCFLAGS += -g
# CCPPFLAGS += -g

all: ${TARGETS}


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
#include ${PETSC_DIR}/conf/rules

############### Set Util
set_util.o: setutil/set_util.cpp
	g++ -c  ${DEBUG} -fPIC -Wall -std=c++11 -o $@ $^

var_list.o: setutil/var_list.cpp
	g++ -c  ${DEBUG} -fPIC -Wall -std=c++11 -o $@ $^

cnpy.o: setutil/cnpy/cnpy.cpp
	g++ -c  ${DEBUG} -fPIC -Wall -std=c++11 -o $@ $^

setutil/set_util.cpp: setutil/set_util.h
setutil/var_list.cpp: setutil/var_list.h

libset_util.a: set_util.o var_list.o
	ar rcs $@ $^

_set_util.so: set_util.o var_list.o
	g++ ${DEBUG} -fPIC -Wall -std=c++11 -shared -Wl,-soname,$@ -o $@ $^

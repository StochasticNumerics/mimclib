TARGETS =  libset_util.so libset_util.a
DEBUG = -O3
# DEBUG = -pg
# CCFLAGS += -g
# CCPPFLAGS += -g

all: ${TARGETS}


#include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules
#include ${PETSC_DIR}/conf/rules

############### Set Util
set_util.o: src/set_util.cpp
	g++ -c  ${DEBUG} -fPIC -Wall -std=c++11 -o $@ $^

var_list.o: src/var_list.cpp
	g++ -c  ${DEBUG} -fPIC -Wall -std=c++11 -o $@ $^


src/set_util.cpp: src/set_util.h
src/var_list.cpp: src/var_list.h

libset_util.a: set_util.o var_list.o
	ar rcs $@ $^

libset_util.so: set_util.o var_list.o
	g++ ${DEBUG} -fPIC -Wall -std=c++11 -shared -Wl,-soname,$@ -o $@ $^

clean:
	${RM} *.so *.a *.o

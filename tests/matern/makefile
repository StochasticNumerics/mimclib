TARGETS = _matern.so
DEBUG = -O3
# DEBUG = -pg
# CCFLAGS += -g
# CCPPFLAGS += -g

all: ${TARGETS}

############### matern
_matern.so: matern.o common.o
	${CLINKER} -fPIC -rdynamic -Wall -shared -Wl,-soname,$@ -o $@  $^ ${PETSC_LIB} -llapack -lblas ${DEBUG}

matern.o: matern.cpp
	${PETSC_COMPILE_SINGLE} -fPIC  -rdynamic $< ${DEBUG}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
#include ${PETSC_DIR}/conf/rules

common.o: common.cpp common.hpp
	${PETSC_COMPILE_SINGLE} $< ${DEBUG}

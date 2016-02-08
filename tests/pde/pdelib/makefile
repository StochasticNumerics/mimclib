TARGETS = libsolver_nd_df.so
#DEBUG = -g -pg

all: ${TARGETS}

libsolver_nd_df.so: common.o nd_df.o libset_util.a
	${CLINKER} -Wall -shared -Wl,-soname,$@ -o $@  $^ ${PETSC_LIB} -llapack -lblas ${DEBUG}

common.o: src/common.c src/common.h
	${PETSC_COMPILE_SINGLE} $< ${DEBUG}

nd_df.o: src/nd_df.c src/SField.h src/common.h
	${PETSC_COMPILE_SINGLE} $< ${DEBUG}

clean::
	-@${RM} *.o ${TARGETS}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
#include ${PETSC_DIR}/conf/rule

CCFLAGS += -g
CCPPFLAGS += -g

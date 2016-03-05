TARGETS = _libwcumsum.so
DEBUG = -O3
# DEBUG = -pg
# CCFLAGS += -g
# CCPPFLAGS += -g

all: ${TARGETS}

wcumsum.o: wcumsum.c
	gcc -c  ${DEBUG} -fPIC -Wall -o $@ $^

_libwcumsum.so: wcumsum.o
	gcc ${DEBUG} -fPIC -Wall -std=c++11 -shared -Wl,-soname,$@ -o $@ $^

clean:
	${RM} *.so *.a *.o

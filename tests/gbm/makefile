TARGETS = libwcumsum.so
# DEBUG = -pg
# CCFLAGS += -g
# CCPPFLAGS += -g

all: ${TARGETS}

libwcumsum.so: wcumsum.c
	gcc ${DEBUG} -fPIC -Wall -shared -O3 -o $@ $^

clean:
	${RM} *.so *.a *.o

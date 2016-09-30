#!/usr/bin/python
if __name__ == "__main__":
    from mimclib.plot import run_program
    from mimclib import ipdb
    ipdb.set_excepthook()
    run_program()

all: clean deluser inplace user

user:
	python setup.py install --user

inplace:
	python setup.py build_ext --inplace

clean:
	rm -rf build dist mimclib.egg-info

deluser:
	rm -rf /home/abdo/.local/lib/python2.7/site-packages/mimclib*

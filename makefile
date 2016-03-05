all: clean deluser inplace user

user:
	python setup.py install --user

inplace:
	python setup.py build_ext --inplace

clean:
	-$(RM) -r build dist mimclib.egg-info

deluser:
	-$(RM) -r $(shell python -m site --user-site)/mimclib*
#-$(RM) -r $(shell python -m site --user-site)/mimclib-*-py*.egg-info

all: clean deluser inplace user

user:
	python setup.py install --user

inplace:
	python setup.py build_ext --inplace

clean:
	python setup.py clean
	-$(RM) -r build dist mimclib.egg-info

# TODO: Maybe we can use pip to deluser?
deluser:
	-$(RM) -r $(shell python -m site --user-site)/mimclib*
#-$(RM) -r $(shell python -m site --user-site)/mimclib-*-py*.egg-info

pip:
	pip install --user -e .[mysqldb]

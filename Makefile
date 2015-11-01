
build:symmetry.pyx
	python setup.py build_ext --inplace

run:build
	python test_symmetry.py

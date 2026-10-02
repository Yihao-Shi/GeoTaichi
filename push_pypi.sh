python setup.py sdist build
python setup.py bdist_wheel
twine upload dist /*

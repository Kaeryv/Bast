pip-package:
	python -m build

publish: pip-package
	scp dist/khepri-0.1.1-py3-none-any.whl stukov:/srv/http/blog/

test:
	python -m unittest discover test/

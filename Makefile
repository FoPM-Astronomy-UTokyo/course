.PHONY: test update

test:
	mkdocs serve

update:
	mkdocs gh-pages

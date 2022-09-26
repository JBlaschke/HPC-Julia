NBCONVERT := jupyter nbconvert --to slides --stdout

all: Julia.html Profiling.html Python.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

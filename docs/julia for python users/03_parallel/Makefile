NBCONVERT := jupyter nbconvert --to slides --stdout

all: Parallel.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

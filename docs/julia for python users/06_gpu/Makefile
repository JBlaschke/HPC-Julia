NBCONVERT := jupyter nbconvert --to slides --stdout

all: CUDA.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

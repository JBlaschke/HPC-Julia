NBCONVERT := jupyter nbconvert --to slides --stdout

all: Automatic_Differentiation.html Flux.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

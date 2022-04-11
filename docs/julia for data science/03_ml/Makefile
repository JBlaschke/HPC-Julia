NBCONVERT := jupyter nbconvert --to slides --stdout

all: Automatic_Differentiation.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

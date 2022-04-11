NBCONVERT := jupyter nbconvert --to slides --stdout

all: Arrays.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

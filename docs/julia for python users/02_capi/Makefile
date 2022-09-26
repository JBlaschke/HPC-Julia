NBCONVERT := jupyter nbconvert --to slides --stdout

all: Calling_C.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

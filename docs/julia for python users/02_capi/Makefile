NBCONVERT := jupyter nbconvert --to slides --stdout

all: Calling_C.html PyCall.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

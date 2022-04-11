NBCONVERT := jupyter nbconvert --to slides --stdout

all: Arrays.html File_IO.html Database_IO.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

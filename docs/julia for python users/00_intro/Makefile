NBCONVERT := jupyter nbconvert --to slides --stdout

all: Getting_Started.html Data_Types.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

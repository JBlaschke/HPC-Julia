NBCONVERT := jupyter nbconvert --to slides --stdout

all: Array-Based_Data_Types.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

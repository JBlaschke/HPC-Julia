NBCONVERT := jupyter nbconvert --to slides --stdout

all: Getting_Started.html Data_Types.html Program_Structure.html Methods_Introspection.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

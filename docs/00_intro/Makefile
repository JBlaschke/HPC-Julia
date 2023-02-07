NBCONVERT := jupyter nbconvert --template my-reveal --to slides --stdout

all: \
	Getting_Started.html       \
	Data_Types.html            \
	Methods_Introspection.html \
	Program_Structure.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

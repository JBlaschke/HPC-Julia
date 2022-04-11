NBCONVERT := jupyter nbconvert --to slides --stdout

all: Plotting.html Statistics.html Interact.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

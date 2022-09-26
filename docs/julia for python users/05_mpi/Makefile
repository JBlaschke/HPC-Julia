NBCONVERT := jupyter nbconvert --to slides --stdout

all: DistributedArrays.html MPIArrays.html OneSided.html SharedArrays.html

%.html: %.ipynb
	$(NBCONVERT) $^ > $@

clean:
	rm *.html

TOPTARGETS := all clean
NBCONVERT  := jupyter nbconvert --to slides --stdout
SUBDIRS    := 00_intro 01_arrays 02_capi 03_parallel 04_numba 05_mpi


all: index.html
	for dir in $(SUBDIRS); \
	do                     \
	    $(MAKE) -C $$dir;  \
	done

%.html: %.ipynb
	$(NBCONVERT) $^ > $@


clean:
	rm *.html
	for dir in $(SUBDIRS);       \
	do                           \
	    $(MAKE) -C $$dir clean;  \
	done


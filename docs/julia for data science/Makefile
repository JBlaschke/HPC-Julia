TOPTARGETS := all clean
NBCONVERT  := jupyter nbconvert --to slides --stdout
SUBDIRS    := 00_intro 01_data 03_ml


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


TOPTARGETS := all clean
NBCONVERT  := jupyter nbconvert --template my-reveal --to slides --stdout
SUBDIRS    := 00_intro


all: presentation_index.html
	for dir in $(SUBDIRS); \
	do                     \
	    $(MAKE) -C $$dir;  \
	done


%.html: %.ipynb
	$(NBCONVERT) $^ > $@


clean:
	rm presentation_index.html;  \
	for dir in $(SUBDIRS);       \
	do                           \
	    $(MAKE) -C $$dir clean;  \
	done


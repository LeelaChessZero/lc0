ifndef EXE
	EXE = $(CURDIR)/lc0
endif

all:
	chmod +x ../build.sh
ifdef EVALFILE
	../build.sh -Dembed=true && mv ../build/release/lc0 $(EXE)
	cat $(EVALFILE) >> $(EXE)
	perl -e "printf '%sLc0!', pack('V', -s '$(EVALFILE)')" >> $(EXE)
else
	../build.sh && mv ../build/release/lc0 $(EXE)
endif
	

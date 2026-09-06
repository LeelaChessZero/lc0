ifndef EXE
	EXE = $(CURDIR)/lc0
endif

BUILD_FLAGS =
ifdef EVALFILE
	BUILD_FLAGS += -Dembed=true
endif
ifdef SEARCH
	BUILD_FLAGS += -Ddefault_search=$(SEARCH)
endif

POST_BUILD_COMMANDS =
ifdef EVALFILE
	POST_BUILD_COMMANDS = \
		cat $(EVALFILE) >> $(EXE); \
		perl -e "printf '%sLc0!', pack('V', -s '$(EVALFILE)')" >> $(EXE)
endif

.PHONY: all
all:
	chmod +x ../build.sh
	../build.sh $(BUILD_FLAGS) && mv ../build/release/lc0 $(EXE)
	$(POST_BUILD_COMMANDS)
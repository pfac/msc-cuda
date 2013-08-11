texmain = $(shell basename `pwd`)
textmpexts = aux bbl bcf blg fdb_latexmk fls out run.xml synctex.gz toc
texoutexts = log pdf

PHONY: clean purge

clean:
	- $(RM) $(textmpexts:%=$(texmain).%)

purge: clean
	- $(RM) $(texoutexts:%=$(texmain).%)

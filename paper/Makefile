all: main.pdf

main.pdf: main.tex bibliography.bib
	pdflatex main.tex
	bibtex main
	pdflatex main.tex


clean:
	rm -f main.bbl main.blg main.out paperNotes2.bib texput2.log main.aux main.log

watch:
	echo "Watching .tex files for changes..."
	watchmedo shell-command --command='make all' -w -p='*.tex'

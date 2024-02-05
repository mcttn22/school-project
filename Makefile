TEX = latexmk
TEXFLAGS = -pdf

all: project_report clean

pdfs:
	for notebook in ./notebooks/*.ipynb; do \
		jupyter nbconvert --LatexPreprocessor.title='' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf $$notebook; \
	done

project_report:
	$(TEX) $(TEXFLAGS) -jobname=./project-report/project-report -shell-escape ./project-report/src/latex/project-report.tex

clean:
	$(TEX) -c -jobname=./project-report/project-report ./project-report/src/latex/project-report.tex
TEX = latexmk
TEXFLAGS = -pdf

all: project_report clean

project_report:
	$(TEX) $(TEXFLAGS) -jobname=./project-report/project-report -shell-escape ./project-report/src/latex/project-report.tex

clean:
	$(TEX) -c -jobname=./project-report/project-report ./project-report/src/latex/project-report.tex
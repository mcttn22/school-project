TEX = latexmk
TEXFLAGS = -pdf

all: docs project_report clean

docs:
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/ann ./docs/models/src/latex/ann.tex
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/maths ./docs/models/src/latex/maths.tex
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/image_recognition/cat ./docs/models/src/image_recognition/latex/cat.tex
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/image_recognition/number ./docs/models/src/image_recognition/latex/number.tex
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/utils/perceptron_model ./docs/models/src/utils/latex/perceptron_model.tex
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/utils/shallow_model ./docs/models/src/utils/latex/shallow_model.tex

project_report:
	$(TEX) $(TEXFLAGS) -jobname=./project-report/project-report ./project-report/src/latex/project-report.tex

clean:
	$(TEX) -c -jobname=./docs/models/ann ./docs/models/src/latex/ann.tex
	$(TEX) -c -jobname=./docs/models/maths ./docs/models/src/latex/maths.tex
	$(TEX) -c -jobname=./docs/models/image_recognition/cat ./docs/models/src/image_recognition/latex/cat.tex
	$(TEX) -c -jobname=./docs/models/image_recognition/number ./docs/models/src/image_recognition/latex/number.tex
	$(TEX) -c -jobname=./docs/models/utils/perceptron_model ./docs/models/src/utils/latex/perceptron_model.tex
	$(TEX) -c -jobname=./docs/models/utils/shallow_model ./docs/models/src/utils/latex/shallow_model.tex
	$(TEX) -c -jobname=./project-report/project-report ./project-report/src/latex/project-report.tex
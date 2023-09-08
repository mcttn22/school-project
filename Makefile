TEX = latexmk
TEXFLAGS = -pdf

all: model_theory write_up clean

model_theory: ann maths image_recognition utils

ann:
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/ann ./docs/models/src/latex/ann.tex

maths:
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/maths ./docs/models/src/latex/maths.tex

image_recognition: cat number

cat:
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/image_recognition/cat ./docs/models/src/image_recognition/latex/cat.tex

number:
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/image_recognition/number ./docs/models/src/image_recognition/latex/number.tex

utils: perceptron_model shallow_model

perceptron_model:
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/utils/perceptron_model ./docs/models/src/utils/latex/perceptron_model.tex

shallow_model:
	$(TEX) $(TEXFLAGS) -jobname=./docs/models/utils/shallow_model ./docs/models/src/utils/latex/shallow_model.tex

write_up:
	$(TEX) $(TEXFLAGS) -jobname=./write-up ./write-up/src/latex/write-up.tex

clean:
	$(TEX) -c -jobname=./docs/models/ann ./docs/models/src/latex/ann.tex
	$(TEX) -c -jobname=./docs/models/maths ./docs/models/src/latex/maths.tex
	$(TEX) -c -jobname=./docs/models/image_recognition/cat ./docs/models/src/image_recognition/latex/cat.tex
	$(TEX) -c -jobname=./docs/models/image_recognition/number ./docs/models/src/image_recognition/latex/number.tex
	$(TEX) -c -jobname=./docs/models/utils/perceptron_model ./docs/models/src/utils/latex/perceptron_model.tex
	$(TEX) -c -jobname=./docs/models/utils/shallow_model ./docs/models/src/utils/latex/shallow_model.tex
	$(TEX) -c -jobname=./write-up ./write-up/src/latex/write-up.tex
TEX = latexmk
TEXFLAGS = -pdf

all: ann maths image_recognition utils clean

ann:
	$(TEX) $(TEXFLAGS) -jobname=./ann src/latex/ann.tex

maths:
	$(TEX) $(TEXFLAGS) -jobname=./maths src/latex/maths.tex

image_recognition: cat number

cat:
	$(TEX) $(TEXFLAGS) -jobname=./image_recognition/cat src/image_recognition/latex/cat.tex

number:
	$(TEX) $(TEXFLAGS) -jobname=./image_recognition/number src/image_recognition/latex/number.tex

utils: perceptron_model shallow_model

perceptron_model:
	$(TEX) $(TEXFLAGS) -jobname=./utils/perceptron_model src/utils/latex/perceptron_model.tex

shallow_model:
	$(TEX) $(TEXFLAGS) -jobname=./utils/shallow_model src/utils/latex/shallow_model.tex

clean:
	$(TEX) -c -jobname=./ann src/latex/ann.tex
	$(TEX) -c -jobname=./maths src/latex/maths.tex
	$(TEX) -c -jobname=./image_recognition/cat src/image_recognition/latex/cat.tex
	$(TEX) -c -jobname=./image_recognition/number src/image_recognition/latex/number.tex
	$(TEX) -c -jobname=./utils/perceptron_model src/utils/latex/perceptron_model.tex
	$(TEX) -c -jobname=./utils/shallow_model src/utils/latex/shallow_model.tex
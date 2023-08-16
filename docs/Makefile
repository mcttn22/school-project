all: ann cat_recognition maths numpy clean

ann:
	latexmk -pdf -jobname=./ann src/latex/ann.tex

cat_recognition:
	latexmk -pdf -jobname=./cat_recognition src/latex/cat_recognition.tex

maths:
	latexmk -pdf -jobname=./extra/maths src/latex/maths.tex

numpy:
	latexmk -pdf -jobname=./extra/numpy src/latex/numpy.tex

clean:
	latexmk -c src/latex/ann.tex
	latexmk -c src/latex/cat_recognition.tex
	latexmk -c -jobname=./extra/maths src/latex/maths.tex
	latexmk -c -jobname=./extra/numpy src/latex/numpy.tex
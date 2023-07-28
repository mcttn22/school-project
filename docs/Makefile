all: maths numpy xor clean

numpy:
	latexmk src/latex/numpy.tex

maths:
	latexmk src/latex/maths.tex

xor: 
	latexmk src/latex/xor_model.tex

image:
	latexmk src/latex/image_model.tex

clean:
	latexmk -c src/latex/maths.tex
	latexmk -c src/latex/numpy.tex
	latexmk -c src/latex/xor_model.tex
	latexmk -c src/latex/image_model.tex
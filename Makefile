TEX = latexmk
TEXFLAGS = -pdf

all: project_report clean

notebooks:
	jupyter nbconvert --LatexPreprocessor.title='Learning Rate Analysis' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/learning-rate-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Epoch Count Analysis' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/epoch-count-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Train Dataset Size Analysis' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/train-dataset-size-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Layer Count Analysis' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/layer-count-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Neuron Count Analysis' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/neuron-count-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='ReLu Analysis' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/relu-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='CPU vs GPU Analysis' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/cpu-vs-gpu-analysis.ipynb;

project_report:
	$(TEX) $(TEXFLAGS) -jobname=./project-report/project-report -shell-escape ./project-report/src/latex/project-report.tex

clean:
	$(TEX) -c -jobname=./project-report/project-report ./project-report/src/latex/project-report.tex
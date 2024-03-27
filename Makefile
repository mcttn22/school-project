TEX = latexmk
TEXFLAGS = -pdf

all: project_report clean

notebook-pdfs:
	jupyter nbconvert --LatexPreprocessor.title='Analysis of the impact of the learning rate on the reduction of the loss value' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/learning-rate-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Analysis of the impact of training epoch count on network performance and training time taken' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/epoch-count-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Analysis of the impact of training dataset size on network performance and training time taken' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/train-dataset-size-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Analysis of the impact of layer count on network performance and training time taken' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/layer-count-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Analysis of the impact of neuron count on network performance and training time taken' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/neuron-count-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Analysis of the impact of the transfer function on the reduction of the loss value' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/relu-analysis.ipynb;
	jupyter nbconvert --LatexPreprocessor.title='Analysis of the impact of using a CPU vs GPU on training time taken' --LatexPreprocessor.date='' --LatexPreprocessor.author_names='' --output-dir='./project-report/src/pdfs' --to pdf ./notebooks/cpu-vs-gpu-analysis.ipynb;

project_report:
	$(TEX) $(TEXFLAGS) -jobname=./project-report/project-report -shell-escape ./project-report/src/latex/project-report.tex

clean:
	$(TEX) -c -jobname=./project-report/project-report ./project-report/src/latex/project-report.tex
	rm ./project-report/project-report.bbl

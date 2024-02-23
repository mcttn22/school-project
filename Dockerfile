FROM python:3.11

# Set a directory for the app
WORKDIR /usr/src/app

# Copy all the files to the container
COPY . .

# Install dependencies
RUN python setup.py install

# Run the project
CMD ["python", "./school_project"]
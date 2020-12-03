# Object Recognition and Computer Vision (First Semester)

# Code
Python3 with jupyter-notebook.

You can install the requirements with

```bash
$ pip install -r requirements.txt
```

For the first assignment, the need of a special library (uneasy to install) made me create a Docker image 
with this library installed alongside jupyter.

This small docker file is improvable. (Specially for loading/saving the notebook which cannot be done easily)

Some useful commands:

```bash
$ # Build the image:
$ docker build . -t MyOwnJupyterImage
$
$ # Run the container
$ docker run -p 8888:8888 -t MyOwnJupyterImage
$
$ # Restart the container (can be useful to retrieve the last version of the notebook)
$ docker start -a $CONTAINER_ID
```

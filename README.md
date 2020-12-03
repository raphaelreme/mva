# mva

## Python environments and jupyter kernels

I'm using `pyenv` and `virtualenvwrapper` to generate different environments for the courses. Even if it's not required, some notebooks will refer to a specific kernel which is unknown on your machine. In that case you can just change the kernel to one of yours.


Some useful bash commands:
```bash
$ # Create a new environment
$ mkvirtualenv [-a DIR] env_name
$
$ # Use the environment (and jump to directory DIR if provided)
$ workon env_name
$
$ # Create a kernel that uses this environment (Work also with conda environment)
$ pip install ipykernel
$ python -m ipykernel install --user --name kernel_name
$
$ # List kernels
$ jupyter-kernelspecs
$
$ # Stop using the environment
$ deactivate
$
$ # Remove the environment
$ rmvirtualenv env_name
$
$ # Delete the associated kernel
$ jupyter-kernelspec remove kernel_name
```

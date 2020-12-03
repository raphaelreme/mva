FROM jupyter/scipy-notebook

# Back to root user to install our very special lib...
USER root

RUN wget -N http://www.di.ens.fr/willow/teaching/recvis_orig/assignment1/install_cyvlfeat.py
RUN python install_cyvlfeat.py

# Use the jupyter notebook user from the base image.
USER $NB_UID

# ENTRYPOINT/CMD are already define in the base image as needed.

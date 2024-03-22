FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential

# hdf5 needed for pytables installation
# libgles2-mesa needed for pytest-qt
RUN apt-get install -y libhdf5-dev libgles2-mesa-dev

RUN python -m pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

CMD ["/bin/bash"]
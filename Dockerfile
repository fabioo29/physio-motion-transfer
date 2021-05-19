# use CUDA + OpenGL
FROM nvidia/cudagl:10.0-devel-ubuntu16.04
MAINTAINER Fabio Oliveira (fabiodiogo29@gmail.com)

# install apt dependencies
RUN apt-get update && apt-get install -y \
	git \
	vim \
	wget \
	software-properties-common \
	curl

# install newest cmake version
RUN apt-get purge cmake && cd ~ && wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz && tar -xvf cmake-3.14.5.tar.gz
RUN cd ~/cmake-3.14.5 && ./bootstrap && make -j6 && make install

# install python3.7 and pip
RUN apt-add-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 && \
    ln -s /usr/bin/python3.7 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py | python

# setting up cudnn
RUN apt-get install -y --no-install-recommends \             
	libcudnn7=7.4.1.5-1+cuda10.0 \             
	libcudnn7-dev=7.4.1.5-1+cuda10.0 
RUN apt-mark hold libcudnn7 && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN python -m pip install tensorflow-gpu==1.14.0

# install dirt
ENV CUDAFLAGS='-DNDEBUG=1'
RUN cd ~ && git clone https://github.com/pmh47/dirt.git && \ 
 	python -m pip install dirt/

# run dirt test command
RUN python ~/dirt/tests/square_test.py

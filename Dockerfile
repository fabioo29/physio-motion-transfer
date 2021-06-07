FROM nvidia/cudagl:10.0-devel-ubuntu16.04
MAINTAINER Fabio Oliveira (fabiodiogo29@gmail.com)

ADD ./pmt $HOME/root/pmt/

# install apt dependencies
RUN apt-get -y --no-install-recommends update && \
	apt-get -y --no-install-recommends upgrade && \
	apt-get install -y --no-install-recommends \
	build-essential \
	git \
	vim \
	wget \
	libatlas-base-dev \
	libprotobuf-dev \
	libleveldb-dev \
	libsnappy-dev \
	libhdf5-serial-dev \
	protobuf-compiler \
	libboost-all-dev \
	libgflags-dev \
	libgoogle-glog-dev \
	liblmdb-dev \
	pciutils \
	python3-setuptools \
	python3-dev \
	python3-pip \
	opencl-headers \
	ocl-icd-opencl-dev \
	libviennacl-dev \
	libcanberra-gtk-module \
	libopencv-dev \
	software-properties-common \
	curl

# install newest cmake version
RUN apt-get purge -y cmake && cd ~ && wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz && tar -xvf cmake-3.14.5.tar.gz
RUN cd ~/cmake-3.14.5 && ./bootstrap && make -j6 && make install

# install python3.7 and pip
RUN rm -rf /usr/bin/python  && \
    apt-add-repository -y ppa:deadsnakes/ppa && \
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
RUN python -m pip install tensorflow-gpu==1.14.0 numpy protobuf opencv-python
RUN apt-get update && apt-get install -y python3.7-dev libpython3.7-dev

# install dirt
ENV CUDAFLAGS='-DNDEBUG=1'
RUN cd ~ && git clone https://github.com/pmh47/dirt.git && \ 
 	python -m pip install dirt/

# run dirt test command
#RUN python ~/dirt/tests/square_test.py

# build pyopenpose
RUN cd ~ && git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git && \
	cd openpose && \
	mkdir build && \
	cd build

RUN cd ~/openpose/build && \
	cmake -DBUILD_PYTHON=ON .. && \
	make -j`nproc` && \
	make -j`nproc` && \
	mv python/openpose/pyopenpose.cpython-37m-x86_64-linux-gnu.so /usr/local/lib/python3.7/dist-packages/

# install pmt requirements
RUN ls ~
RUN cd ~/pmt/ && pip install -r requirements.txt

# add pmt large files to respective dirs
RUN apt install -y megatools unzip && cd /tmp/ && megadl 'https://mega.nz/#!sOhmwQbT!IICjPAEy-uzcnQNaAZC2nl77SGUp-BnYmil-cSVNP8s' && unzip pmt-large-files.zip

RUN cd /tmp/ && \
	mv model.ckpt-593292.data-00000-of-00001 ~/pmt/thirdparty/cihp_pgn/assets/ && \
	cp neutral_smpl.pkl  ~/pmt/thirdparty/octopus/assets/ && \
	mv octopus_weights.hdf5  ~/pmt/thirdparty/octopus/assets/ && \
	mv pose_iter_584000.caffemodel  ~/pmt/thirdparty/octopus/assets/pose/body_25/ && \
	mv pose_iter_116000.caffemodel  ~/pmt/thirdparty/octopus/assets/face && \
	mv basicModel_f_lbs_10_207_0_v1.0.0.pkl ~/pmt/thirdparty/romp/assets/ && \
	mv basicmodel_m_lbs_10_207_0_v1.0.0.pkl ~/pmt/thirdparty/romp/assets/ && \
	mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ~/pmt/thirdparty/romp/assets/ && \
	mv ROMP_hrnet32.pkl ~/pmt/thirdparty/romp/assets/ && \
	mv ROMP_hrnet32+CAR.pkl ~/pmt/thirdparty/romp/assets/ && \
	mv ROMP_resnet50.pkl ~/pmt/thirdparty/romp/assets/ && \
	mv SMPL_NEUTRAL.pkl ~/pmt/thirdparty/romp/assets/ && \
	mv dp_uv_lookup_256.npy ~/pmt/thirdparty/tex2shape/assets/ && \
	mv tex2shape_weights.hdf5 ~/pmt/thirdparty/tex2shape/assets/ && \
	mv neutral_smpl.pkl ~/pmt/thirdparty/tex2shape/assets/

# install opendr
RUN cd ~ && git clone https://github.com/yifita/opendr.git && cd opendr/ && pip install . && \
	python setup.py build && python setup.py install && pip install .

# install caffe2 & pytorch
# add MAX_JOBS=2 before python setup.py install if install error
RUN git clone https://github.com/pytorch/pytorch.git && cd pytorch && \
	pip install -r https://raw.githubusercontent.com/pytorch/pytorch/master/requirements.txt && \
	pip install pyyaml==5.4.1 && \
	git submodule update --init --recursive && \
	python setup.py install

# install cocoapi
RUN git clone https://github.com/cocodataset/cocoapi.git && \
	make install && \
	python setup.py install --user

# install densepose python3
RUN git clone https://github.com/stimong/densepose_python3.git densepose && \
	cd densepose && \ pip install -r requirements.txt && \
	pip install opencv-python==4.2.0.32 && \
	make && \
	ln -s  /root/densepose/build/lib.linux-x86_64-3.7/detectron/utils/* /usr/local/lib/python3.7/dist-packages/

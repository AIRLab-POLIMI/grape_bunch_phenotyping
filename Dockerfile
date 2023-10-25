FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# ********************************************************
# * Set the second group name below *
# ********************************************************

ARG GROUP2_NAME=phd
ARG GROUP2_ID=20002

# ********************************************************
# 
# ********************************************************

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	build-essential ca-certificates python3.8 python3.8-dev python3.8-distutils git wget cmake unzip libboost-python-dev
RUN ln -sv /usr/bin/python3.8 /usr/bin/python

# Create a non-root user
RUN groupadd -g $USER_GID $USERNAME \
	# [Optional] Remove the following line if you don't want to use the second group. Remove also the "--groups $GROUP2_NAME" option in the useradd command below.
	&& groupadd -g $GROUP2_ID $GROUP2_NAME \ 
	&& useradd -u $USER_UID -g $USER_GID --groups $GROUP2_NAME -m $USERNAME \
	#
	# [Optional] Add sudo support. Omit if you don't need to install software after connecting.
	&& apt-get install -y sudo \
	&& echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py --user && \
	rm get-pip.py

# install dependencies
RUN pip install --user opencv-python
RUN pip install --user matplotlib
RUN pip install --user albumentations
RUN pip install --user tensorboard
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
RUN pip install --user torch-tb-profiler
RUN pip install --user neptune-client
RUN pip install --user optuna

RUN pip install --user onnx 
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# install linters
RUN pip install --user flake8
RUN pip install --user pylint

# install the Neptune-Optuna integration
RUN pip install --user neptune-optuna
RUN pip install --user shapely
RUN pip install --user torcheval
RUN pip install --user openpyxl
RUN pip install --user seaborn
RUN pip install --user nbformat
RUN pip install --user statsmodels

RUN git clone https://github.com/ricber/pyodi
WORKDIR /home/$USERNAME/pyodi
RUN pip install --user .
WORKDIR /home/$USERNAME

RUN pip install --user fiftyone==0.21.6

# SORT dependencies
RUN pip install --user filterpy==1.4.5
RUN pip install --user lap==0.4.0

# muSSP dependencies
ENV PYTHONPATH="/home/$USERNAME/mot3d:${PYTHONPATH}"
RUN pip install --user numpy
RUN pip install --user networkx
RUN pip install --user scipy
RUN pip install --user pyyaml
RUN pip install --user imageio
RUN pip install --user tqdm
RUN pip install --user PuLP==2.4
# muSSP installation
RUN git clone https://github.com/cvlab-epfl/mot3d.git
WORKDIR /home/$USERNAME/mot3d
RUN unzip mussp-master-6cf61b8.zip
WORKDIR /home/$USERNAME/mot3d/mot3d/solvers/wrappers/muSSP
RUN ./cmake_and_build.sh
WORKDIR /home/$USERNAME

# COLMAP
USER root
RUN apt-get install -y \
	ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
USER $USERNAME
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /home/$USERNAME/colmap
RUN mkdir build
WORKDIR /home/$USERNAME/colmap/build
RUN cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=all-major
RUN ninja
USER root
RUN ninja install
USER $USERNAME

CMD ["bash"]
WORKDIR /home/$USERNAME
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install APT tools and repos
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    apt-utils \
    software-properties-common 

RUN apt-get update && apt-get upgrade -y

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install system tools
RUN apt-get install -y \
    git \
    net-tools \
    wget \
    curl \
    zip \
    unzip \
    patchelf \
    python3-pip

# #Install SOFA Dependencies
RUN apt install -y \
    gcc-10 \
    cmake \
    qt5-default \
    libboost-all-dev \
    python3-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libglew-dev \
    zlib1g-dev \
    libeigen3-dev

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy scipy pybind11

RUN mkdir /opt/sofa
RUN git clone --depth 1 -b v22.12 https://github.com/sofa-framework/sofa.git /opt/sofa/src

RUN mkdir /opt/sofa/build


RUN cmake -D SOFA_FETCH_SOFAPYTHON3=True -D SOFA_FETCH_BEAMADAPTER=True -D PYTHON_VERSION=3.8 -D pybind11_DIR=/usr/local/lib/python3.8/dist-packages/pybind11/share/cmake/pybind11/ -D PYTHON_EXECUTABLE=/usr/bin/python3.8 -G "CodeBlocks - Unix Makefiles" -S /opt/sofa/src -B /opt/sofa/build
RUN cmake -D PLUGIN_SOFAPYTHON3=True -D PLUGIN_BEAMADAPTER=True -S /opt/sofa/src -B /opt/sofa/build

RUN make -j 4 --directory /opt/sofa/build
RUN make install --directory /opt/sofa/build

ENV PYTHONPATH="/opt/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages/:PYTHONPATH"
ENV SOFA_ROOT="/opt/sofa/build/install"

RUN python3 -m pip install nvidia-cuda-cupti-cu11==11.7.101
RUN python3 -m pip install nvidia-cublas-cu11==11.10.3.66
RUN python3 -m pip install nvidia_cudnn_cu12==8.9.2.26
RUN python3 -m pip install nvidia-cufft-cu11==10.9.0.58
RUN python3 -m pip install nvidia-curand-cu11==10.2.10.91
RUN python3 -m pip install nvidia-cusolver-cu11==11.4.0.1
RUN python3 -m pip install nvidia-cusparse-cu11==11.7.4.91
RUN python3 -m pip install nvidia-nccl-cu11==2.14.3


RUN python3 -m pip install torch==2.1.0
RUN python3 -m pip install torchvision torchaudio
RUN python3 -m pip install scipy scikit-image pyvista PyOpenGL pygame matplotlib pillow opencv-python meshio pyyaml optuna gymnasium

RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

COPY . /opt/eve_training
RUN python3 -m pip install /opt/eve_training/eve
RUN python3 -m pip install -e /opt/eve_training/eve_bench
RUN python3 -m pip install /opt/eve_training/eve_rl
# RUN python3 -m pip install /opt/eve_training

WORKDIR /opt/eve_training
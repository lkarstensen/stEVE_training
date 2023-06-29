FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04
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
RUN python3 -m pip install nvidia-cudnn-cu11==8.5.0.96
RUN python3 -m pip install nvidia-cufft-cu11==10.9.0.58
RUN python3 -m pip install nvidia-curand-cu11==10.2.10.91
RUN python3 -m pip install nvidia-cusolver-cu11==11.4.0.1
RUN python3 -m pip install nvidia-cusparse-cu11==11.7.4.91
RUN python3 -m pip install nvidia-nccl-cu11==2.14.3


RUN python3 -m pip install torch   
RUN python3 -m pip install torchvision torchaudio
RUN python3 -m pip install scipy scikit-image pyvista PyOpenGL pygame matplotlib pillow opencv-python meshio pyyaml optuna gymnasium

RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

COPY . /opt/eve_training
RUN python3 -m pip install /opt/eve_training/eve
RUN python3 -m pip install /opt/eve_training/eve_bench
RUN python3 -m pip install /opt/eve_training/eve_rl
RUN python3 -m pip install /opt/eve_training

WORKDIR /opt/eve_training

# docker buildx build --platform=linux/amd64 -t 10.15.17.136:5555/lnk/eve_training -f ./dockerfile .
# docker push 10.15.17.136:5555/lnk/eve_training
# docker pull 10.15.17.136:5555/lnk/eve_training

# docker buildx build --platform=linux/amd64 -t 10.15.17.136:5555/lnk/eve_training -f ./dockerfile . && docker push 10.15.17.136:5555/lnk/eve_training

# docker container stop $(docker container ls --filter label=lnk_training --quiet) ; docker pull 10.15.17.136:5555/lnk/eve_training

# docker image rm $(docker image ls --filter reference="registry.gitlab.cc-asp.fraunhofer.de/lnk/eve_registry" --filter "dangling=true" --quiet)

# .73
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/cerebral/aorta/optuna_heatup.py -d cuda -nw 55 -n archgen

# .197
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/cerebral/aorta/optuna_heatup.py -d cuda -nw 29 -n archgen
#./eve_training/eve_paper/cerebral/aorta/train_arch_generator1.py -d cuda -nw 29 -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -n archgen

# .223
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/lnk_training/results --shm-size 15G -d registry.gitlab.cc-asp.fraunhofer.de/stacie/ma_projects/lnk_training:mac python3 ./lnk_training/ijcars23v2/seed_only.py -d cuda -nw 20 -n seeds_only -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -nsv 128

# .238
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/cerebral/aorta/optuna_heatup.py -d cuda -nw 20 -n archgen

# .123
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/eve_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/cerebral/aorta/optuna_heatup.py -d cuda -nw 20 -n archgen

#.17.164
# docker run --label=lnk_training --gpus all --mount type=bind,source=$PWD/results,target=/opt/lnk_training/results --shm-size 15G -d registry.gitlab.cc-asp.fraunhofer.de/stacie/ma_projects/lnk_training:mac python3 ./lnk_training/ijcars23v2/seed_only.py -d cuda -nw 14 -n ff_seeds_only -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -ff -nsv 128

# pamb-dlp
# docker run --label=lnk_training --gpus all --rm --mount type=bind,source=$PWD/results,target=/opt/lnk_training/results --shm-size 15G -d 10.15.17.136:5555/lnk/eve_training python3 ./eve_training/eve_paper/cerebral/aorta/optuna_heatup.py -d cuda -nw 65 -n archgen
# docker run --label=lnk_training --gpus all --rm --mount type=bind,source=$PWD/results,target=/opt/lnk_training/results --shm-size 15G -d registry.gitlab.cc-asp.fraunhofer.de/stacie/ma_projects/lnk_training:mac python3 ./lnk_training/ijcars23v2/seed_only.py -d cuda:1 -nw 35 -n seeds_only -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -nsv 32


# 2023-06-15_13-29_archgen

# 2023-06-15_132954_archgen
# parameters
ARG ARCH
ARG NAME
ARG ORGANIZATION
ARG DESCRIPTION
ARG MAINTAINER

# ==================================================>
# ==> Do not change the code below this line
ARG BASE_REGISTRY=docker.io
ARG BASE_ORGANIZATION=cbschaff
ARG BASE_REPOSITORY=cpk-isaac-gym
ARG BASE_TAG=main

# define base image
FROM ${BASE_REGISTRY}/${BASE_ORGANIZATION}/${BASE_REPOSITORY}:${BASE_TAG}-${ARCH} as BASE

# recall all arguments
# - current project
ARG NAME
ARG ORGANIZATION
ARG DESCRIPTION
ARG MAINTAINER
# - base project
ARG BASE_REGISTRY
ARG BASE_ORGANIZATION
ARG BASE_REPOSITORY
ARG BASE_TAG
# - defaults
ARG LAUNCHER=default

# define/create project paths
ARG PROJECT_PATH="${CPK_SOURCE_DIR}/${NAME}"
ARG PROJECT_LAUNCHERS_PATH="${CPK_LAUNCHERS_DIR}/${NAME}"
RUN mkdir -p "${PROJECT_PATH}"
RUN mkdir -p "${PROJECT_LAUNCHERS_PATH}"
WORKDIR "${PROJECT_PATH}"

# keep some arguments as environment variables
ENV \
    CPK_PROJECT_NAME="${NAME}" \
    CPK_PROJECT_DESCRIPTION="${DESCRIPTION}" \
    CPK_PROJECT_MAINTAINER="${MAINTAINER}" \
    CPK_PROJECT_PATH="${PROJECT_PATH}" \
    CPK_PROJECT_LAUNCHERS_PATH="${PROJECT_LAUNCHERS_PATH}" \
    CPK_LAUNCHER="${LAUNCHER}"

# install apt dependencies
COPY ./dependencies-apt.txt "${PROJECT_PATH}/"
RUN cpk-apt-install ${PROJECT_PATH}/dependencies-apt.txt

# install python3 dependencies
COPY ./dependencies-py3.txt "${PROJECT_PATH}/"
RUN cpk-pip3-install ${PROJECT_PATH}/dependencies-py3.txt

# install launcher scripts
COPY ./launchers/. "${PROJECT_LAUNCHERS_PATH}/"
COPY ./launchers/default.sh "${PROJECT_LAUNCHERS_PATH}/"
RUN cpk-install-launchers "${PROJECT_LAUNCHERS_PATH}"

# copy project root
COPY ./*.cpk ./*.sh ${PROJECT_PATH}/

# copy the source code
COPY ./packages "${CPK_PROJECT_PATH}/packages"

# build catkin workspace
RUN catkin build \
    --workspace ${CPK_CODE_DIR}

# define default command
CMD ["bash", "-c", "launcher-${CPK_LAUNCHER}"]

# store module metadata
LABEL \
    cpk.label.current="${ORGANIZATION}.${NAME}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.description="${DESCRIPTION}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.code.location="${PROJECT_PATH}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.registry="${BASE_REGISTRY}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.organization="${BASE_ORGANIZATION}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.project="${BASE_REPOSITORY}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.tag="${BASE_TAG}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.maintainer="${MAINTAINER}"
# <== Do not change the code above this line
# <==================================================


ENV DISPLAY=:10
ENV QT_X11_NO_MITSHM=1

# NOTE: uncomment this block if you want to install the nvidia drivers inside the container
ENV NVIDIA_DRIVER_VERSION=470.103.01
RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run
# ENV NVIDIA_DRIVER_VERSION=470.103.01
# RUN wget https://us.download.nvidia.com/tesla/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run
RUN chmod +x ./NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run
RUN ./NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run --accept-license --no-questions --ui=none --skip-depmod --no-drm --no-kernel-module

# nvidia runtime configuration
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# copy assets
RUN mkdir /assets
COPY assets/xorg.nvidia.conf /assets/xorg.nvidia.conf

# install mujoco and dm_control
RUN mkdir /root/.mujoco \
    && cd /root/.mujoco \ 
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz \
    && tar -xvzf mujoco-2.1.1-linux-x86_64.tar.gz \
    && rm mujoco-2.1.1-linux-x86_64.tar.gz \
    && python -m pip install dm_control==0.0.425341097

# install pytorch3d
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

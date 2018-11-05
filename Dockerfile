FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

MAINTAINER UNN CG_3

ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

RUN apt-get update 
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get install -y python3.6
RUN apt install -y git pkg-config
RUN apt install -y wget python3-pip python3.6-dev net-tools libtool ccache unzip unrar tar xz-utils bzip2 gzip coreutils ntp
RUN apt-get install -y vim htop nano mc cmake
RUN apt-get clean -y
RUN pip3 install tensorflow-gpu
USER root
# Copy code
ADD ./ /root/QD_classifier
RUN cd /root/QD_classifier && pip3 install -r requirements.txt

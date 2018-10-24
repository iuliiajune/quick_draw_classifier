FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

MAINTAINER UNN CG_3

ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

RUN apt install -y git pkg-config
RUN apt install -y wget python-pip python-dev net-tools libtool ccache unzip unrar tar xz-utils bzip2 gzip coreutils ntp
RUN apt-get install -y vim htop nano mc cmake ffmpeg
RUN apt-get clean -y

USER root
# Copy code
ADD ./ /root/QD_classifier
RUN cd /root/DeepSpeech && pip3 install -r requirements.txt

# Copy models
ADD build/models/ /root/DeepSpeech/inference/models
RUN cd /root/DeepSpeech/inference/server/proto_generated && python run_codegen.py


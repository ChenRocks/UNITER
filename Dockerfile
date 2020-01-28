FROM nvcr.io/nvidia/pytorch:19.05-py3

# basic python packages
RUN pip install pytorch-pretrained-bert==0.6.2 \
                tensorboardX==1.7 ipdb==0.12 lz4==2.1.9 lmdb==0.97

####### horovod for multi-GPU (distributed) training #######

# update OpenMPI to avoid horovod bug
RUN rm -r /usr/local/mpi &&\ 
    wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.4.tar.gz &&\
    gunzip -c openmpi-3.1.4.tar.gz | tar xf - &&\
    cd openmpi-3.1.4 &&\
    ./configure --prefix=/usr/local/mpi --enable-orterun-prefix-by-default \
        --with-verbs --disable-getpwuid &&\
    make -j$(nproc) all && make install &&\
    ldconfig &&\
    cd - && rm -r openmpi-3.1.4 && rm openmpi-3.1.4.tar.gz

ENV OPENMPI_VERSION=3.1.4

# horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod==0.16.4 &&\
    ldconfig

# ssh
RUN apt-get update &&\
    apt-get install -y --no-install-recommends openssh-client openssh-server &&\
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


WORKDIR /src

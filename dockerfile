FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
# ^ CUDA 9.2 works both with PyTorch 0.4.1 and 1.2

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion unzip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh \
 && /bin/bash ~/anaconda.sh -b -p /opt/conda  \
 && rm ~/anaconda.sh  \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh  \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN conda create -y -n py36_py041 python=3.6  \
 && /opt/conda/bin/conda install -n py36_py041 -y pytorch=0.4.1 cuda92  -c pytorch \
 && /opt/conda/bin/conda install -n py36_py041 -y tqdm -c conda-forge \
 && /opt/conda/bin/conda install -n py36_py041 -y six future \
 && conda create -y -n py36_py121 python=3.6  \
 && /opt/conda/bin/conda install -n py36_py121 -y pytorch torchvision cudatoolkit=9.2 ignite -c pytorch \
 && /opt/conda/bin/conda install -n py36_py121 -y rdkit=2019.03.4.0 -c rdkit \
 && /opt/conda/bin/conda install -n py36_py121 -y tqdm -c conda-forge \
 && /opt/conda/bin/conda install -n py36_py121 -y pytest arrow ipython future Pillow seaborn \
 && /opt/conda/bin/conda install -n py36_py121 -y scipy==1.1.0 \
 && /opt/conda/bin/conda clean --all

RUN /opt/conda/envs/py36_py041/bin/pip install --no-cache-dir torchtext==0.3.1

RUN /opt/conda/envs/py36_py121/bin/pip install --no-cache-dir  tabulate \
    tensorboard \
    multiset \
    dataclasses \
    docopt \
    lazy \
 && /opt/conda/envs/py36_py121/bin/pip install  --no-cache-dir git+https://github.com/PatWalters/rd_filters.git \
 && /opt/conda/envs/py36_py121/bin/pip install  --no-cache-dir guacamol

RUN apt-get update --fix-missing && apt-get install -y vim \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pschwllr/MolecularTransformer.git /molecular_transformer
COPY molecular_transformer_weights.pt /molecular_transformer/molecular_transformer_weights.pt
# ^ see README to find details of how to download these weights

COPY . /molecule_chef
RUN unzip -o /molecule_chef/data.zip -d /molecule_chef

WORKDIR /molecule_chef
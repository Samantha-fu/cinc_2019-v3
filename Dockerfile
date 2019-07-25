#FROM python:3.7.3-stretch
FROM nvidia/cuda:8.0-cudnn6-devel-centos7
## The MAINTAINER instruction sets the Author field of the generated images
##MAINTAINER fms@sample.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

# You can use alternative base mirror from https://hub.docker.com/r/nvidia/cuda
#MAINTAINER Will_Ye "jiajie.ye@hotmail.com"

# 安装你程序需要用到的所有依赖项，如Python，numpy，tensorflow等等
RUN set -ex \
    && yum install -y wget tar libffi-devel zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make initscripts \
    && wget https://www.python.org/ftp/python/3.5.0/Python-3.5.0.tgz \
    && tar -zxvf Python-3.5.0.tgz \
    && cd Python-3.5.0 \
    && ./configure prefix=/usr/local/python3 \
    && make \
    && make install \
    && make clean \
    && cd .. \
    && rm -rf /Python-3.5.0* \
    && yum install -y epel-release \
    && yum install -y python-pip
RUN set -ex \
    # 备份旧版本python
    && mv /usr/bin/python /usr/bin/python27 \
    && mv /usr/bin/pip /usr/bin/pip-python2.7 \
    # 配置默认为python3
    && ln -s /usr/local/python3/bin/pip3 /usr/bin/pip \
    && pip install scipy \  #如果要用到scipy这个包，就需要用python2.7安装，python3.5安装会失败
    && ln -s /usr/local/python3/bin/python3.5 /usr/bin/python \
# 修复因修改python版本导致yum失效问题
RUN set -ex \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/bin/yum \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/libexec/urlgrabber-ext-down \
    && yum install -y deltarpm
RUN yum -y install python-devel scipy
RUN pip install --upgrade pip
RUN pip install matplotlib
RUN pip install --upgrade setuptools
RUN pip install tensorflow-gpu
RUN pip install Pillow
#RUN pip install moviepy
RUN pip install keras
RUN pip install cmake
#安装opencv的这一段有点问题，我还没解决，因为后来发现写的这个版本程序不需要用到cv2，暂时搁置，如果之后解决了，再重新补充，问题出在unzip上，可能要补充安装解压文件的工具就行了，还没试
#RUN set -ex \
#    &&wget https://github.com/opencv/opencv/archive/2.4.13.zip \
#    &&unzip opencv-2.4.13.zip \
#    &&cd opencv-2.4.13 \
#    &&cmake CMakeLists.txt \
#    &&mkdir build \
#    &&cd build \
#    &&cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_FFMPEG=OFF -D CMAKE_INSTALL_PREFIX=/usr/local .. \
#     && make \
#     && make install \
RUN pip install wave
RUN pip install  scikit-image
RUN pip install lightgbm
# Add your project file
#注意这里的路径是相对路径，前面的是本地文件，后面的参数是目标存储路径，指镜像中
#ADD ./competition/application.py /data/application.py
#ADD ./competition/model_weights20190430.h5 /data/model_weights20190430.h5

# Define the entry process command
#这个CMD操作只能有一个，要注意这点


## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
#RUN pip install -r requirements.txt

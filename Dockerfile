FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

RUN apt-get upgrade -y \
    && apt install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install python3.9 -y \ 
    && apt install python3.9-distutils -y

WORKDIR /TrafficSense

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
RUN python3 -m pip install --upgrade setuptools pip

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN python3 -m pip install -r requirements.txt

COPY . .

CMD [ "python3.9" , "detect.py" ]
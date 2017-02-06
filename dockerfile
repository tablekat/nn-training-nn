# FROM ubuntu:15.04
# FROM b.gcr.io/tensorflow/tensorflow
FROM gw000/keras:1.1.1-py2-tf-cpu


ENV foo /testing
COPY . $foo
WORKDIR ${foo}
#ADD . $foo

RUN bash
#RUN python dnn.py


#------------------

### Create docker machine with hyperv:
#docker-machine.exe -D create -d hyperv --hyperv-virtual-switch "New Virtual Switch" --hyperv-memory 1024 default
# ^-- if error for out of memory, it means CANT GET ENOUGH MEMORY FROM HOST, SO LOWER HYPERV-MEMORY
# ^-- had to make virtual switch with https://docs.docker.com/machine/drivers/hyper-v/

# run once for new terminal:
# @FOR /f "tokens=*" %i IN ('docker-machine env default') DO @%i

# run for each update/build:
# docker build -t literally-what .
# docker run -it literally-what bash

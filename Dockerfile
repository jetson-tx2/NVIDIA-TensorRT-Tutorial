FROM nvcr.io/nvidia/tensorrt:19.03-py3

RUN pip install jupyterlab scikit-image

RUN bash /opt/tensorrt/python/python_setup.sh
RUN pip install keras

ADD . /workspace/optimization

WORKDIR /workspace/optimization

CMD ["bash"]

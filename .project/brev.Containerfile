FROM nvcr.io/nvidia/rapidsai/notebooks:25.04-cuda12.8-py3.12

WORKDIR /opt/project/build/

SHELL ["/bin/bash", "-c"]

USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo

RUN echo "ubuntu	ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/workbench

ENV NVWB_UID=1000

ENV NVWB_GID=1000

ENV NVWB_USERNAME=ubuntu

USER $NVWB_USERNAME

COPY --chown=$NVWB_UID:$NVWB_GID  ["preBuild.bash", "/opt/project/build/"]

RUN ["/bin/bash", "/opt/project/build/preBuild.bash"]

USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget

RUN dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; wget -O- https://github.com/tianon/gosu/releases/download/1.17/gosu-${dpkgArch} | install /dev/stdin /usr/local/bin/gosu

COPY  --chmod=755 ["entrypoint.sh", "/"]

ENV NVWB_BASE_ENV_ENTRYPOINT=/home/rapids/entrypoint.sh

USER $NVWB_USERNAME

USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    gh \
    jq \
    vim

USER $NVWB_USERNAME

COPY --chown=$NVWB_UID:$NVWB_GID  ["requirements.txt", "/opt/project/build/"]

RUN /opt/conda/bin/pip install --user \
    -r /opt/project/build/requirements.txt 

COPY --chown=$NVWB_UID:$NVWB_GID  ["postBuild.bash", "/opt/project/build/"]

RUN ["/bin/bash", "/opt/project/build/postBuild.bash"]

USER $NVWB_USERNAME

WORKDIR /project

EXPOSE 8888

ENTRYPOINT ["/entrypoint.sh"]

CMD ["tail", "-f", "/dev/null"]
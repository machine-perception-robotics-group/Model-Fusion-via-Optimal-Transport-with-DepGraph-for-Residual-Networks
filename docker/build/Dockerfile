FROM  pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev vim graphviz sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir jupyter seaborn scikit-learn pot numpy==1.23.4 torchinfo graphviz tensorboard

# 参考: dockerでvolumeをマウントしたときのファイルのowner問題（https://qiita.com/yohm/items/047b2e68d008ebb0f001）
RUN apt-get update \
    && apt-get -y install gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

WORKDIR /

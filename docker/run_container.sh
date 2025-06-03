proj_dir=$(cd $(dirname $0);cd ..;pwd)
echo proj_dir $proj_dir

docker run --rm -it --gpus all --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --name improve_otfusion \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -v $proj_dir/data/improve_otfusion:/workspace/improve_otfusion \
    improve_otfusion:20250204 \
    bash

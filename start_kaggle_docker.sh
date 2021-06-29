# kaggle の docker を起動するシェルスクリプト
# 通常はvscodeのReoprn in Containerを使ったほうが便利

export NAME_IMAGE=kaggle-gpu

# kaggle の docker コンテナが存在していた場合起動する。なければコンテナを建てる。
if [ "$(docker ps -qa -f name="${NAME_IMAGE}")" ]; then
    echo "Image ${NAME_IMAGE} already exist."
    docker start ${NAME_IMAGE}
else
    docker run -itd  --gpus all  -p 8888:8888  -v $PWD/kaggle:/home/kaggle  -w /home/kaggle  --name ${NAME_IMAGE}  -h host  kaggle/python-gpu-build /bin/bash
fi

docker exec -it ${NAME_IMAGE} /bin/bash
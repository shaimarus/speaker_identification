---------------------------------------
docker-compose build
docker-compose up

(надо загрузить модель, загрузится после запуска docker-compose up)
docker commit af070bb1ccfc  speechbrain_get_info_from_audio:latest



------------------------------------------------
docker save -o api_get_info_from_audio.tar api_get_info_from_audio:latest

docker load -i speechbrain.tar

docker compose -f docker-compose_offline.yml up -d







надо отдельно установить hnswlib

Bindings installation
You can install from sources:

apt-get install -y python-setuptools python-pip
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
pip install .




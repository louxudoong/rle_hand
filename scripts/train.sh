set -x

CONFIG=$1
EXPID=${2:-"test_rle"}
CKPT=${3:-"./weights/freihand-laplace-rle.pth"}
PORT=${4:-23456}

HOST=$(hostname -i)

python ./scripts/train.py \
    --nThreads 16 \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --seed 123123 \
    --checkpoint ${CKPT} \

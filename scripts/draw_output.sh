set -x

# cfg path + weights patth,like:
# ./scripts/validate.sh ./configs/256x192_res50_regress-flow.yaml ./weights/coco-laplace-rle.pth
CONFIG=$1
CKPT=${2:-"./weights/freihand-laplace-rle.pth"}
EXPID=${3:-"test_rle"}
PORT=${4:-23456}

HOST=$(hostname -i)

python ./scripts/draw_output.py \
    --nThreads 16 \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --seed 123123 \
    --checkpoint ${CKPT} \

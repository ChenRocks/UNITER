# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1
ANN_DIR=$2

set -e

URL='https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data'
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'
MISSING=$BLOB/ann/missing_nlvr2_imgs.json
if [ ! -f $ANN_DIR/missing_nlvr2_imgs.json ]; then
    wget $MISSING -O $ANN_DIR/missing_nlvr2_imgs.json
fi

for SPLIT in 'train' 'dev' 'test1'; do
    if [ ! -f $ANN_DIR/$SPLIT.json ]; then
        echo "downloading ${SPLIT} annotations..."
        wget $URL/$SPLIT.json -O $ANN_DIR/$SPLIT.json
    fi

    echo "preprocessing ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        -w /src chenrocks/uniter \
        python prepro.py --annotation /ann/$SPLIT.json \
                         --missing_imgs /ann/missing_nlvr2_imgs.json \
                         --output /txt_db/nlvr2_${SPLIT}.db --task nlvr2
done

echo "done"

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1 
ANN_DIR=$2 # where refercoco annotations are saved

set -e

if [ ! -f $ANN_DIR/iid2bb_id/iid_to_ann_ids.json ]; then
    echo "pre-compute iid_to_ann_ids.json for all RE datasets following https://github.com/lichengunc/MAttNet/blob/butd_feats/tools/map_iid_to_ann_ids.py ..."
    exit
fi

for DATA in 'refcoco' 'refcoco+'; do
    for SPLIT in 'train' 'val' 'testA' 'testB'; do
        echo "preprocessing ${DATA} ${SPLIT} annotations..."
        docker run --ipc=host --rm -it \
            --mount src=$(pwd),dst=/src,type=bind \
            --mount src=$OUT_DIR,dst=/txt_db,type=bind \
            --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
            -w /src chenrocks/uniter \
            python prepro.py --annotation /ann/$DATA/refs\(unc\).p /ann/$DATA/instances.json /ann/iid2bb_id/iid_to_ann_ids.json \
                            --task re \
                            --output /txt_db/${DATA}_${SPLIT}.db
    done
done

DATA='refcocog'
for SPLIT in 'train' 'val' 'test'; do
    echo "preprocessing ${DATA} ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        -w /src chenrocks/uniter \
        python prepro.py --annotation /ann/$DATA/refs\(umd\).p /ann/$DATA/instances.json /ann/iid2bb_id/iid_to_ann_ids.json \
                        --task re \
                        --output /txt_db/${DATA}_${SPLIT}.db
done

echo "done"

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'

# image dbs
for SPLIT in 'train2014' 'val2014'; do
    if [ ! -d $DOWNLOAD/img_db/coco_$SPLIT ] ; then
        wget $BLOB/img_db/coco_$SPLIT.tar -P $DOWNLOAD/img_db/
        tar -xvf $DOWNLOAD/img_db/coco_$SPLIT.tar -C $DOWNLOAD/img_db
    fi
done
if [ ! -d $DOWNLOAD/img_db/vg ] ; then
    wget $BLOB/img_db/vg.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/vg.tar -C $DOWNLOAD/img_db
fi

# text dbs
for SPLIT in 'train' 'restval' 'val'; do
    wget $BLOB/txt_db/pretrain_coco_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/pretrain_coco_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done
for SPLIT in 'train' 'val'; do
    wget $BLOB/txt_db/pretrain_vg_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/pretrain_vg_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

# converted BERT
for MODEL in base large; do
    if [ ! -f $DOWNLOAD/pretrained/uniter-$MODEL-init.pt ] ; then
        wget $BLOB/pretrained/uniter-$MODEL-init.pt -P $DOWNLOAD/pretrained/
    fi
done


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
for SPLIT in 'train2014' 'val2014' 'test2015'; do
    if [ ! -d $DOWNLOAD/img_db/coco_$SPLIT ] ; then
        wget $BLOB/img_db/coco_$SPLIT.tar -P $DOWNLOAD/img_db/
        tar -xvf $DOWNLOAD/img_db/coco_$SPLIT.tar -C $DOWNLOAD/img_db
    fi
done
wget $BLOB/img_db/vg.tar -P $DOWNLOAD/img_db/
tar -xvf $DOWNLOAD/img_db/vg.tar -C $DOWNLOAD/img_db

# text dbs
for SPLIT in 'train' 'trainval' 'devval' 'test' 'vg'; do
    wget $BLOB/txt_db/vqa_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/vqa_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    wget $BLOB/pretrained/uniter-base.pt -P $DOWNLOAD/pretrained/
fi


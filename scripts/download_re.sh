# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'

# image db
if [ ! -d $DOWNLOAD/img_db/re_coco_gt ] ; then
    wget $BLOB/img_db/re_coco_gt.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/re_coco_gt.tar -C $DOWNLOAD/img_db
fi
if [ ! -d $DOWNLOAD/img_db/re_coco_det ] ; then
    wget $BLOB/img_db/re_coco_det.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/re_coco_det.tar -C $DOWNLOAD/img_db
fi

# text dbs
wget $BLOB/txt_db/re_txt_db.tar -P $DOWNLOAD/txt_db/
tar -xvf $DOWNLOAD/txt_db/re_txt_db.tar -C $DOWNLOAD/txt_db/

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    wget $BLOB/pretrained/uniter-base.pt -P $DOWNLOAD/pretrained/
fi


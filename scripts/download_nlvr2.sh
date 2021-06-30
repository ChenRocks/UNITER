# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'ann' 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'

# annotations
NLVR='https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data'
wget $NLVR/dev.json -P $DOWNLOAD/ann/
wget $NLVR/test1.json -P $DOWNLOAD/ann/

# image dbs
for SPLIT in 'train' 'dev' 'test'; do
    wget $BLOB/img_db/nlvr2_$SPLIT.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/nlvr2_$SPLIT.tar -C $DOWNLOAD/img_db
done

# text dbs
for SPLIT in 'train' 'dev' 'test1'; do
    wget $BLOB/txt_db/nlvr2_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/nlvr2_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    wget $BLOB/pretrained/uniter-base.pt -P $DOWNLOAD/pretrained/
fi

wget $BLOB/finetune/nlvr-base.tar -P $DOWNLOAD/finetune/
tar -xvf $DOWNLOAD/finetune/nlvr-base.tar -C $DOWNLOAD/finetune

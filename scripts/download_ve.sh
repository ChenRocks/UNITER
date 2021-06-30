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
if [ ! -d $DOWNLOAD/img_db/flickr30k ] ; then
    wget $BLOB/img_db/flickr30k.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/flickr30k.tar -C $DOWNLOAD/img_db
fi

# text dbs
for SPLIT in 'train' 'dev' 'test'; do
    wget $BLOB/txt_db/ve_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/ve_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    wget $BLOB/pretrained/uniter-base.pt -P $DOWNLOAD/pretrained/
fi


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'

for MODEL in uniter-base uniter-large; do
    # This will overwrite models
    wget $BLOB/pretrained/$MODEL.pt -O $DOWNLOAD/pretrained/$MODEL.pt
done

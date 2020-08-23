# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://convaisharables.blob.core.windows.net/uniter'

for MODEL in uniter-base uniter-large; do
    wget $BLOB/pretrained/$MODEL.pt -P $DOWNLOAD/pretrained/
done

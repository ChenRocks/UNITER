# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://convaisharables.blob.core.windows.net/uniter'


wget $BLOB/pretrained/uniter-base.pt -P $DOWNLOAD/pretrained/uniter-base.pt

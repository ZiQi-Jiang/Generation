#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
mkdir -p data/imagenet/images

(
    cd data/imagenet/images

    wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
    wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

    mkdir -p train val

    tar -xf ILSVRC2012_img_train.tar -C train

    for archive in train/*.tar; do
        class_dir="${archive%.tar}"
        mkdir -p "$class_dir"
        tar -xf "$archive" -C "$class_dir"
        rm "$archive"
    done

    tar -xf ILSVRC2012_img_val.tar -C val

    wget -O valprep.sh \
        https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

    (
        cd val
        bash ../valprep.sh
    )
)
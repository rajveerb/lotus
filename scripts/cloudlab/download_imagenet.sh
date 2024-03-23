# !/bin/bash
train_dataset_dir="/mydata/P3Tracer/imagenet/train"
mkdir -P $train_dataset_dir
echo "Downloading ImageNet train dataset (please be patient, this may take a while)"
wget --quiet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -P $train_dataset_dir
pushd $train_dataset_dir
echo "Extracting ImageNet train dataset"
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
popd
echo "Finished ImageNet train dataset setup!"

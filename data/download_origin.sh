#!/bin/bash

. ../utils/parse_yaml.sh
. ../utils/gdownload.sh
. ../utils/conditional.sh

eval $(parse_yaml ../config.yml)
echo 'this is the data_path you are trying to download data into:'
echo $data_path
mycwd=$(pwd)

cd $data_path

# this section is for downloading the original CUB dataset
# credit to http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# md5sum for the downloaded CUB_200_2011.tgz should be 97eceeb196236b17998738112f37df78
echo "downloading CUB..."
gdownload 1hbzc_P1FuxMkcabkgn9ZKinBwW683j45 CUB_200_2011.tgz
conditional_targz CUB_200_2011.tgz 97eceeb196236b17998738112f37df78
if [ -d "CUB_200_2011" ] 
then
    rm attributes.txt
fi


# this section is for downloading the original Aircraft dataset
# credit to http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
# md5sum for the downloaded fgvc-aircraft-2013b.tar.gz should be d4acdd33327262359767eeaa97a4f732
echo "downloading FGVC-Aircraft..."
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
conditional_targz fgvc-aircraft-2013b.tar.gz d4acdd33327262359767eeaa97a4f732


# this section is for downloading images for mini-ImageNet
# credit to https://github.com/mileyan/simple_shot, https://github.com/twitter/meta-learning-lstm
# md5sum for the downloaded images.zip should be 987d2dfede486f633ec052ff463b62c6
echo "downloading images for mini-ImageNet..."
gdownload 0B3Irx3uQNoBMQ1FlNXJsZUdYWEE images.zip
conditional_unzip images.zip 987d2dfede486f633ec052ff463b62c6


# this section is for downloading images for tiered-ImageNet in the original version
# credit to https://github.com/renmengye/few-shot-ssl-public
# md5sum for the downloaded tiered-imagenet.tar should be e07e811b9f29362d159a9edd0d838c62
echo "downloading images for tiered-ImageNet..."
gdownload 1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07 tiered-imagenet.tar
conditional_tar tiered-imagenet.tar e07e811b9f29362d159a9edd0d838c62


# this section is for downloading images for tiered-ImageNet data used in DeepEMD
# credit to https://github.com/icoz69/DeepEMD/blob/master/datasets/download_tieredimagenet.sh
# md5sum for the downloaded tiered_imagenet.tar should be 7828a6dc2889e226ba575d2ba9624753
echo "downloading images for tiered-ImageNet_DeepEMD..."
gdownload 1ANczVwnI1BDHIF65TgulaGALFnXBvRfs tiered_imagenet.tar
conditional_tar tiered_imagenet.tar 7828a6dc2889e226ba575d2ba9624753
if [ -d "tiered_imagenet" ] 
then
    mv tiered_imagenet tiered-ImageNet_DeepEMD
fi


# this section is for downloading images for meta-iNat and tiered meta-iNat
# credit to https://github.com/visipedia/inat_comp/tree/master/2017
# md5sum for the downloaded train_val_images.tar.gz should be 7c784ea5e424efaec655bd392f87301f
# md5sum for the downloaded train_2017_bboxes.zip should be 761c954e69bf23d87c5d0d0c68d79bd5
echo "downloading images for meta-iNat / tiered meta-iNat..."
wget -q --show-progress https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz
conditional_targz train_val_images.tar.gz 7c784ea5e424efaec655bd392f87301f
if [ -d "train_val_images" ]
then
    mkdir inat2017
    mv train_val_images inat2017/train_val_images
    cd inat2017
    wget -q --show-progress https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip
    conditional_unzip train_2017_bboxes.zip 761c954e69bf23d87c5d0d0c68d79bd5
fi
cd $mycwd

echo "Downloads complete!"

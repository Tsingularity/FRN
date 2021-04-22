#!/bin/bash

. ../utils/parse_yaml.sh
. ../utils/gdownload.sh
. ../utils/conditional.sh

eval $(parse_yaml ../config.yml)
echo 'this is the data_path you are trying to download data into:'
echo $data_path

cd $data_path

# this section is for downloading the CUB_fewshot_cropped
# md5sum for the downloaded CUB_fewshot_cropped.tar should be 3f74dd5afdf8b38559bf1c303a16075f
echo "downloading CUB_fewshot_cropped..."
gdownload  1Fp2ItDQfWmb6YsaOtCTFwDLtTvLhWDYA CUB_fewshot_cropped.tar
conditional_tar CUB_fewshot_cropped.tar 3f74dd5afdf8b38559bf1c303a16075f

# this section is for downloading the CUB_fewshot_raw
# md5sum for the downloaded CUB_fewshot_raw.tar should be 914db78ae781778d04d321ad9e9c19da
echo "downloading CUB_fewshot_raw..."
gdownload 10GV9XJYyNL1uzuC04s9fANsQDz9LEj4f CUB_fewshot_raw.tar
conditional_tar CUB_fewshot_raw.tar 914db78ae781778d04d321ad9e9c19da

# this section is for downloading the Aircraft_fewshot
# md5sum for the downloaded Aircraft_fewshot.tar should be f6646b79d6223af5de175c4849eecf25
echo "downloading Aircraft_fewshot..."
gdownload 1BynNYtQM1i8Rv5_4i52CWhwgSWDwjVkc Aircraft_fewshot.tar
conditional_tar Aircraft_fewshot.tar f6646b79d6223af5de175c4849eecf25

# this section is for downloading meta-iNat and tiered meta-iNat
# md5sum for the downloaded meta_inat.tar should be 01482d47c863005a048b0909f86fcaae
echo "downloading meta-iNat / tiered meta-iNat ..."
gdownload 1ATRnifcZ7-_7YxXbKbCi6mL61hfk_YEJ meta_inat.tar
conditional_tar meta_inat.tar 01482d47c863005a048b0909f86fcaae

# this section is for downloading the mini-ImageNet
# md5sum for the downloaded mini-ImageNet.tar should be 13fda464dcd4d283e953bfb6633176b8
echo "downloading mini-ImageNet..."
gdownload 1MfEd5MZlgO6lhrigCaKfxxLAsUoaDtMw mini-ImageNet.tar
conditional_tar mini-ImageNet.tar 13fda464dcd4d283e953bfb6633176b8

# this section is for downloading the tiered-ImageNet
# md5sum for the downloaded tiered-ImageNet.tar should be d127a0635b02446c120f2bd6036832c7
echo "downloading tiered-ImageNet..."
gdownload 11khUm-kfRBcgUzkJi5v0fpt9IT8CdKaR tiered-ImageNet.tar
conditional_tar tiered-ImateNet.tar d127a0635b02446c120f2bd6036832c7

# this section is for downloading the tiered-ImageNet_DeepEMD
# md5sum for the downloaded tiered-ImageNet_DeepEMD.tar should be 1527ba3454a8d28b5ded2fe6df12e82c
echo "downloading tiered-ImageNet_DeepEMD..."
gdownload 1lnqFmOuXuFLGiSRSeluWHCl_NOUXw1Hc tiered-ImageNet_DeepEMD.tar
conditional_tar tiered-ImageNet.tar 1527ba3454a8d28b5ded2fe6df12e82c

echo ""
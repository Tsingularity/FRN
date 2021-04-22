#!/bin/bash

. ../utils/gdownload.sh
. ../utils/conditional.sh

# this section is for downloading the trained_model_weights.tar
# md5sum for the downloaded trained_model_weights.tar should be 0dde61f0b520ae9c17c9aae6dcc64b55
echo "downloading trained_model_weights..."
gdownload 1n6zg5Bkj5FzSHpRbJdzlUkSVDtmjgobm trained_model_weights.tar
conditional_tar trained_model_weights.tar 0dde61f0b520ae9c17c9aae6dcc64b55
echo "trained_model_weights downloaded"
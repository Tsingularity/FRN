#!/bin/bash

conditional_tar () {
    mds=$(md5sum $1 | grep '[0-9,a-f]\{32\}' -o)
    if [[ $mds = $2 ]] ; then
        echo "Download successful! Extracting..."
        tar -xf $1 --checkpoint=.2000
    else
        echo "Download error. Try manually downloading and then extract."
        rm $1
    fi
}

conditional_targz () {
    mds=$(md5sum $1 | grep '[0-9,a-f]\{32\}' -o)
    if [[ $mds = $2 ]] ; then
        echo "Download successful! Extracting..."
        tar -xzf $1 --checkpoint=.2000
    else
        echo "Download error. Try manually downloading and then extract."
        rm $1
    fi
}

conditional_unzip () {
    mds=$(md5sum $1 | grep '[0-9,a-f]\{32\}' -o)
    if [[ $mds = $2 ]] ; then
        echo "Download successful! Extracting..."
        unzip $1 | awk 'BEGIN {ORS=" "} {if(NR%2000==0)print "."}'
    else
        echo "Download error. Try manually downloading and then extract."
        rm $1
    fi
}
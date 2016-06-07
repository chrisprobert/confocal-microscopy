#!/usr/bin/env bash

for i in $(seq 0 9); do
    wget "http://images.yeastrc.org/imagerepo/download/full_download/full_download_${i}.tgz"
    tar -xzf full_download_${i}.tgz
done;

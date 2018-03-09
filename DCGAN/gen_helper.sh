#!/bin/bash

ROOT_PATH="/home/lsw/Workspace/EGRID_GAN/DCGAN/tensorboard/pair"
TIME="2018-03-09T15:02:02.391359"
CONFIG="$ROOT_PATH/config-$TIME"
FOLDER="$ROOT_PATH/ann-$TIME"
DEVICE=3
N_SAMPLES=10

echo Generate griddata from $CONFIG ...
python gen.py --config="$CONFIG" --n_samples=$N_SAMPLES --device=$DEVICE --interpolate

echo Make visit inputs...
cd $FOLDER
python ~/bin/expand.py *.grid
make_visit *.grid

echo Move visit inputs ...
mkdir visit
mv *.bov *.times visit

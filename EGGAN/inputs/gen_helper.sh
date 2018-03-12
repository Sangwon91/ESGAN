#!/bin/bash

ROOT_PATH="/home/lsw/Workspace/EGRID_GAN/EGGAN/tensorboard/pair"
TIME="2018-03-12T12:02:51.752591"
CONFIG="$ROOT_PATH/config-$TIME"
FOLDER="$ROOT_PATH/ann-$TIME"
DEVICE=3
N_SAMPLES=100000

echo Generate griddata from $CONFIG ...
python gen.py --config="$CONFIG" --n_samples=$N_SAMPLES --device=$DEVICE #--interpolate

#echo Make visit inputs...
#cd $FOLDER
#python ~/bin/expand.py *.grid
#make_visit *.grid
#
#echo Move visit inputs ...
#mkdir visit
#mv *.bov *.times visit

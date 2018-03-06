#!/bin/bash

TIME="2018-03-02T18:37:26.734375"
#CONFIG="logdir/ortho/std/config-$TIME"
#FOLDER="logdir/ortho/std/ann-$TIME"
CONFIG="logdir/march/config-$TIME"
FOLDER="logdir/march/ann-$TIME"
INFER_CONFIG="/home/lsw/Workspace/EGRID_GAN/DCGAN/tensorboard/test/config-2018-03-03T09:59:39.369058"
DEVICE=3
N_SAMPLES=100000

echo Generate griddata from $CONFIG ...
python gen.py --config="$CONFIG" --n_samples=$N_SAMPLES --device=$DEVICE

echo Start cell parameter inferences...
python infer_cell_resnet.py --config="$INFER_CONFIG" --griddata_folder="$FOLDER" --device=$DEVICE
#python infer.py --config="$INFER_CONFIG" --griddata_folder="$FOLDER" --device=$DEVICE

#echo Make visit inputs...
#cd $FOLDER
#make_visit *.grid

#echo Move visit inputs ...
#mkdir visit
#mv *.bov *.times visit

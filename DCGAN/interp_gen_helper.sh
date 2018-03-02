#!/bin/bash

TIME="2018-02-27T21:12:05.591534"
CONFIG="logdir/ortho/std/config-$TIME"
FOLDER="logdir/ortho/std/ann-$TIME"
INFER_CONFIG="logdir/ortho/frac2cell/config-2018-02-21T00:01:05.552610"
DEVICE=3
N_SAMPLES=10

echo Generate griddata from $CONFIG ...
python interpolate.py --config="$CONFIG" --n_samples=$N_SAMPLES --device=$DEVICE

echo Start cell parameter inferences...
python infer.py --config="$INFER_CONFIG" --griddata_folder="$FOLDER" --device=$DEVICE

echo Make visit inputs...
cd $FOLDER
python ~/bin/expand.py *.grid
make_visit *.grid

echo Move visit inputs ...
mkdir visit
mv *.bov *.times visit

#!/usr/bin/env bash

python -m modules.para_selection.DataHelper --batch_size 5 --working_place $1 --n_cpus 4
#python -m para_selection.para_selector --batch_size 5 --working_place local --n_cpus 4
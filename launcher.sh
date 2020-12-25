#!/usr/bin/env bash

python -m para_selection.DataHelper --task selectparas_createData --batch_size 5 --working_place local --n_cpus 4
#python -m para_selection.para_selector --batch_size 5 --working_place local --n_cpus 4
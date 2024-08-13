#!/bin/bash

combos='lw_rp_h'

for combo in $combos
do 
  echo Running combo $combo
  # python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserGlobalModel' --fast_dev_run
  python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoser_LwRpH_Model'
done

#!/bin/bash


cd nonclosure
python y_binned_observables.py 

cd ../fullscan_stat
python y_binned_observables.py


cd ../fullscan1
python y_binned_observables.py 1
cd ../fullscan2
python y_binned_observables.py 2
cd ../fullscan3
python y_binned_observables.py 3
cd ../fullscan4
python y_binned_observables.py 4
cd ../fullscan5
python y_binned_observables.py 5
cd ../fullscan6
python y_binned_observables.py 6
cd ../fullscan7
python y_binned_observables.py 7
cd ../fullscan8
python y_binned_observables.py 8




#!/bin/bash


cd nonclosure
python get_Q2_spectrum.py 

cd ../fullscan_stat
python get_Q2_spectrum.py


cd ../fullscan1
python get_Q2_spectrum.py 1
cd ../fullscan2
python get_Q2_spectrum.py 2
cd ../fullscan3
python get_Q2_spectrum.py 3
cd ../fullscan4
python get_Q2_spectrum.py 4
cd ../fullscan5
python get_Q2_spectrum.py 5
cd ../fullscan6
python get_Q2_spectrum.py 6
cd ../fullscan7
python get_Q2_spectrum.py 7
cd ../fullscan8
python get_Q2_spectrum.py 8




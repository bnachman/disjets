import os
import numpy as np

Q2_full_bin = np.logspace(np.log10(100),np.log10(100000),5)
np.save("Q2_full_bin.npy",Q2_full_bin)



#for i in range(1,9):
    #os.system("python fullscan"+str(i)+"/Q2_binned_observables.py "+str(i))
#os.system("python fullscan_stat/Q2_binned_observables.py")
#os.system("python nonclosure/Q2_binned_observables.py")

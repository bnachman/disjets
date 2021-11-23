import os
import numpy as np

Q_cuts = [150,200,250,48000]
np.save("Q_cuts.npy",Q_cuts)

pT_bin = np.logspace(np.log10(10),np.log10(100),7)
np.save("pT_bin.npy",pT_bin)


for i in range(1,9):
    os.system("cp -r fullscan"+str(i)+"/storage_files/ fullscan"+str(i)+"/storage_files_old_pT_bin")
os.system("cp -r fullscan_stat/storage_files/ fullscan_stat/storage_files_old_pT_bin")
os.system("cp -r nonclosure/storage_files/ nonclosure/storage_files_old_pT_bin")


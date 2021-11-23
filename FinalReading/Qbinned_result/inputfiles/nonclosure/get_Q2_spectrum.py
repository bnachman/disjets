import sys


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from matplotlib import gridspec
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


import os
os.environ['CUDA_VISIBLE_DEVICES']="1" #"1"

physical_devices = tf.config.list_physical_devices('GPU') 

tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Django_nominal.pkl")
theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()
pass_unknown_reco = np.array(data['pass_reco'])
theta_unknown_S = theta_unknown_S[pass_unknown_reco==1]
weights_unknown = np.array(data['wgt'])
weights_unknown = weights_unknown[pass_unknown_reco==1]

scaler_data = StandardScaler()
scaler_data.fit(theta_unknown_S)


mc_name = ['Rapgap','Rapgap','Rapgap','Rapgap','Rapgap','Rapgap'] 
mc_tag_out = ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']

mc_name += ['Django','Django','Django','Django','Django','Django'] 
mc_tag_out += ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']


iteration = 4

    
for i in tqdm(range(1)):
    mc = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/"+mc_name[i]+"_"+mc_tag_out[i]+".pkl")

    theta0_G = mc[['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta','genjet_phi','genjet_dphi','genjet_qtnorm','gen_Q2']].to_numpy()
    weights_MC_sim = mc['wgt']
    #p_xyz = mc[["Q2"]].to_numpy()
    pass_reco = np.array(mc['pass_reco'])
    pass_truth = np.array(mc['pass_truth'])
    pass_fiducial = np.array(mc['pass_fiducial'])

    Q2 = theta0_G[:,8]
    Q2 = Q2[pass_fiducial==1]
    theta0_G = theta0_G[:,0:8]

    del mc
    gc.collect()

    NNweights_step2 = np.ones(len(theta0_G))
    for run_iter in range(5):

        mymodel = tf.keras.models.load_model("/global/home/users/yxu2/disjets/FinalReading/inputfiles/nonclosure/models/"+mc_name[i]+"_"+mc_tag_out[i]+"_NC_iteration"+str(run_iter), compile=False)
        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),batch_size=10000)
        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
        NNweights_step2_hold = NNweights_step2_hold[:,0]
        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
        NNweights_step2_hold[pass_truth==0] = 1.
        NNweights_step2 = NNweights_step2_hold*NNweights_step2
            

    ###
    # Q^2
    ###
    bin_full = np.load("/global/home/users/yxu2/disjets/FinalReading/Qbinned_result/inputfiles/Q2_full_bin.npy")
    
    fig = plt.figure(figsize=(7, 5)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")
    ax0.minorticks_on()
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=20)

    n_MC,_,_=plt.hist(Q2,bins=bin_full,weights=weights_MC_sim[pass_fiducial==1],density=True,histtype="step",color="black",label="MC")
    n_Omni_step2_Q2,_,_=plt.hist(Q2,bins=bin_full,weights=weights_MC_sim[pass_fiducial==1]*NNweights_step2[pass_fiducial==1],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

    plt.ylabel("Normalized to unity",fontsize=20)
    plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
    plt.text(0.05, 1.25,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
    plt.legend(frameon=False,fontsize=15)
    plt.locator_params(axis='x', nbins=5)
    plt.ylim([0,0.8])

    ax1 = plt.subplot(gs[1])
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(direction="in",which="both")
    ax1.minorticks_on()
    plt.yscale("log")
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.xlabel("Particle-level $Q^{2}$",fontsize=15)
    plt.ylabel("step 2/step 1",fontsize=15)


    fig.savefig("storage_plots/"+mc_name[i]+"_"+mc_tag_out[i]+"Iteration"+str(iteration)+"_Step2_Q2_NC.pdf",bbox_inches='tight')
    np.save("storage_files_old_pT_bin/"+mc_name[i]+"_"+mc_tag_out[i]+"n_Omni_step2_Q2_iteration"+str(iteration)+"_NC_",n_Omni_step2_Q2)
    
    ###
    # Q^2 (zoomed to 100-1000 GeV^2)
    ###
    bin_zoom = np.logspace(np.log10(100),np.log10(1000),20)
    
    fig = plt.figure(figsize=(7, 5)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")
    ax0.minorticks_on()
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=20)

    n_MC,_,_=plt.hist(Q2,bins=bin_zoom,weights=weights_MC_sim[pass_fiducial==1],density=True,histtype="step",color="black",label="MC")
    n_Omni_step2_Q2,_,_=plt.hist(Q2,bins=bin_zoom,weights=weights_MC_sim[pass_fiducial==1]*NNweights_step2[pass_fiducial==1],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

    plt.ylabel("Normalized to unity",fontsize=20)
    plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
    plt.text(0.05, 1.25,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
    plt.legend(frameon=False,fontsize=15)
    plt.locator_params(axis='x', nbins=5)
    plt.ylim([0,0.8])

    ax1 = plt.subplot(gs[1])
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(direction="in",which="both")
    ax1.minorticks_on()
    plt.yscale("log")
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.xlabel("Particle-level $Q^{2}$",fontsize=15)
    plt.ylabel("step 2/step 1",fontsize=15)


    fig.savefig("storage_plots/"+mc_name[i]+"_"+mc_tag_out[i]+"Iteration"+str(iteration)+"_Step2_Q2_zoomed_NC.pdf",bbox_inches='tight')
    np.save("storage_files_old_pT_bin/"+mc_name[i]+"_"+mc_tag_out[i]+"n_Omni_step2_Q2_zoomed_iteration"+str(iteration)+"_NC_",n_Omni_step2_Q2)

 
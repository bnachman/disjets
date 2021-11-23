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

data = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Data_nominal.pkl")
theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()

scaler_data = StandardScaler()
scaler_data.fit(theta_unknown_S)

bins = {}

#jet pt
bins[0] = np.load("/global/home/users/yxu2/disjets/FinalReading/Qbinned_result/inputfiles/pT_bin.npy")

#jet eta
bins[1] = np.linspace(-1,2.5,6)

#dphi
bins[2] = np.logspace(np.log10(0.03),np.log10(np.pi/2.0),9) - 0.03
bins[2] = bins[2][1:]
bins[2][0] = 0.0

#qt
bins[3] = np.logspace(np.log10(0.03),np.log10(3.03),9) - 0.03
bins[3] = bins[3][1:]
bins[3][0] = 0.0


run_number = sys.argv[1]  #

mc_name = ['Rapgap','Rapgap','Rapgap','Rapgap','Rapgap','Rapgap'] 
mc_tag_out = ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']

mc_name += ['Django','Django','Django','Django','Django','Django'] 
mc_tag_out += ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']

cuts = np.load("/global/home/users/yxu2/disjets/FinalReading/Qbinned_result/inputfiles/Q_cuts.npy")
iteration = 4

    
for i in tqdm(range(12)):
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
        mymodel = tf.keras.models.load_model("/global/home/users/yxu2/disjets/FinalReading/inputfiles/fullscan"+str(run_number)+"/models/"+mc_name[i]+"_"+mc_tag_out[i]+"_iteration"+str(run_iter)+"_step2", compile=False)
        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),batch_size=10000)
        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
        NNweights_step2_hold = NNweights_step2_hold[:,0]
        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
        NNweights_step2_hold[pass_truth==0] = 1.
        NNweights_step2 = NNweights_step2_hold*NNweights_step2
            
    for k in tqdm(range(len(cuts)-1)):
        Q2_cut = np.where((cuts[k] < Q2) & (Q2 < cuts[k+1]), True, False)
        cut_label = str(cuts[k])+" < Q2 < "+str(cuts[k+1]) #+ " (GeV^2)"

        ###
        # eta
        ###

        fig = plt.figure(figsize=(7, 5)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=20)

        n_MC,_,_=plt.hist(theta0_G[pass_fiducial==1][:,4][Q2_cut],bins=bins[1],weights=weights_MC_sim[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",label="MC")
        n_Omni_step2_eta,_,_=plt.hist(theta0_G[pass_fiducial==1][:,4][Q2_cut],bins=bins[1],weights=weights_MC_sim[pass_fiducial==1][Q2_cut]*NNweights_step2[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

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
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

        plt.xlabel("Particle-level $\eta^{jet}$",fontsize=15)
        plt.ylabel("step 2/step 1",fontsize=15)



        fig.savefig("storage_plots/"+mc_name[i]+"_"+mc_tag_out[i]+"Iteration"+str(iteration)+cut_label+"_Step2_eta.pdf",bbox_inches='tight')
        np.save("storage_files/"+mc_name[i]+"_"+mc_tag_out[i]+"n_Omni_step2_eta_iteration"+str(iteration)+cut_label,n_Omni_step2_eta)

        ###
        # pT
        ###

        fig = plt.figure(figsize=(7, 5)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=20)

        n_MC,_,_=plt.hist(theta0_G[pass_fiducial==1][:,3][Q2_cut],bins=bins[0],weights=weights_MC_sim[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",label="MC")
        n_Omni_step2_pT,_,_=plt.hist(theta0_G[pass_fiducial==1][:,3][Q2_cut],bins=bins[0],weights=weights_MC_sim[pass_fiducial==1][Q2_cut]*NNweights_step2[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

        plt.ylabel("Normalized to unity",fontsize=20)
        plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
        plt.text(0.05, 1.25,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
        plt.legend(frameon=False,fontsize=15)
        plt.locator_params(axis='x', nbins=5)
        plt.yscale("log")

        ax1 = plt.subplot(gs[1])
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction="in",which="both")
        ax1.minorticks_on()
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

        plt.xlabel("Particle-level $p_T^{jet}$",fontsize=15)
        plt.ylabel("step 2/step 1",fontsize=15)
        fig.savefig("storage_plots/"+mc_name[i]+"_"+mc_tag_out[i]+"Iteration"+str(iteration)+cut_label+"_Step2_pT.pdf",bbox_inches='tight')
        np.save("storage_files/"+mc_name[i]+"_"+mc_tag_out[i]+"n_Omni_step2_pT_iteration"+str(iteration)+cut_label,n_Omni_step2_pT)

        ###
        # dphi
        ###

        fig = plt.figure(figsize=(7, 5)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=20)

        n_MC,_,_=plt.hist(theta0_G[pass_fiducial==1][:,6][Q2_cut],bins=bins[2],weights=weights_MC_sim[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",label="MC")
        n_Omni_step2_dphi,_,_=plt.hist(theta0_G[pass_fiducial==1][:,6][Q2_cut],bins=bins[2],weights=weights_MC_sim[pass_fiducial==1][Q2_cut]*NNweights_step2[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

        plt.ylabel("Normalized to unity",fontsize=20)
        plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
        plt.text(0.05, 1.25,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
        plt.legend(frameon=False,fontsize=15)
        plt.locator_params(axis='x', nbins=5)
        plt.yscale("log")

        ax1 = plt.subplot(gs[1])
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction="in",which="both")
        ax1.minorticks_on()
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

        plt.xlabel("Particle-level $\Delta\phi$ [rad]",fontsize=15)
        plt.ylabel("step 2/step 1",fontsize=15)

 

        fig.savefig("storage_plots/"+mc_name[i]+"_"+mc_tag_out[i]+"Iteration"+str(iteration)+cut_label+"_Step2_dphi.pdf",bbox_inches='tight')
        np.save("storage_files/"+mc_name[i]+"_"+mc_tag_out[i]+"n_Omni_step2_dphi_iteration"+str(iteration)+cut_label,n_Omni_step2_dphi)


        ###
        # qT
        ###

        fig = plt.figure(figsize=(7, 5)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=20)

        n_MC,_,_=plt.hist(theta0_G[pass_fiducial==1][:,7][Q2_cut],bins=bins[3],weights=weights_MC_sim[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",label="MC")
        n_Omni_step2_qT,_,_=plt.hist(theta0_G[pass_fiducial==1][:,7][Q2_cut],bins=bins[3],weights=weights_MC_sim[pass_fiducial==1][Q2_cut]*NNweights_step2[pass_fiducial==1][Q2_cut],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

        plt.ylabel("Normalized to unity",fontsize=20)
        plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
        plt.text(0.05, 1.25,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
        plt.legend(frameon=False,fontsize=15)
        plt.locator_params(axis='x', nbins=5)
        plt.yscale("log")

        ax1 = plt.subplot(gs[1])
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction="in",which="both")
        ax1.minorticks_on()
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

        plt.xlabel("Particle-level $q_T/Q$",fontsize=15)
        plt.ylabel("step 2/step 1",fontsize=15)

     
        fig.savefig("storage_plots/"+mc_name[i]+"_"+mc_tag_out[i]+"Iteration"+str(iteration)+cut_label+"_Step2_qT.pdf",bbox_inches='tight')
        np.save("storage_files/"+mc_name[i]+"_"+mc_tag_out[i]+"n_Omni_step2_qT_iteration"+str(iteration)+cut_label,n_Omni_step2_qT)

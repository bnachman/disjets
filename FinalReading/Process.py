import gc

#path = '/data1/bpnachman/july16/'
#path = '/clusterfs/ml4hep/bpnachman/H1/hera/'
path = '/clusterfs/ml4hep/yxu2/voltan_all_files/july16/'

mc_names = ['Rapgap','Rapgap','Rapgap','Rapgap','Rapgap','Rapgap'] 
mc_tags = ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']
mc_tags_out = ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']

mc_names += ['Django','Django','Django','Django','Django','Django'] 
mc_tags += ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']
mc_tags_out += ['nominal','sys_0','sys_1','sys_5','sys_7','sys_11']

import os
import uproot3
import pandas as pd
import numpy as np



def get_Dataframe(path, name='Data', tag=None, verbose=False):
    #Files = os.listdir(path) 
    #print (Files)
    df = None
    
    index = np.load("root_index_files/"+mc_name+"_"+tag+"_root_index.npy")
    for i in index:
        
        filename = path+mc_name+"_Eplus0607_"+str(i)+"."+tag+".root"
        if (verbose):
            print ('filename is' , filename)
        
        temp_file = uproot3.open(filename)
        
        hasTree = False 

        if (verbose):
            print (temp_file.keys()) 
                    
        if(len(temp_file.keys())<1):
            if (verbose):
                print('could not find %s, skipping'%name)
            continue
        
        if( not(name in str(temp_file.keys()[0]))):
            if (verbose):
                print('could not find %s, skipping'%name)
            continue
        
        for key in temp_file[name].keys():
            #print (key)
            if('minitree' in str(key)):
                hasTree=True
        if (not hasTree):
            if (verbose):
                print('file has not minitree, skipping')
            continue

        temp_tree = temp_file[name+'/minitree']

        
        temp_df = None
        
        if 'Data' not in name:
            try:
                temp_df   =  temp_tree.pandas.df(["jet*", "genjet*","Q2","gen_Q2","y",'gen_y',"e_*","gene*",'wgt','vertex_z','ptmiss','ptratio*','Empz','pth'], entrystop=3e7,flatten=True)
                df = pd.concat([df,temp_df])
            except ValueError:
                if (verbose):
                    print ('oops, there is a problem in flattening the TTree ')
        else:
            try:
                temp_df   =  temp_tree.pandas.df(["jet*","Q2","y","e_*",'wgt','vertex_z','ptmiss','ptratio*','Empz','pth'], entrystop=3e7,flatten=True) 
                df = pd.concat([df,temp_df])
            except ValueError:
                if (verbose):
                    print ('oops, there is a problem in flattening the TTree ')
        
        #try:
        #    df.shape[0]
        #except ValueError:
        #    print('no valid dataframe')
    if (verbose):
        print('####################################################################')
        if( not(df is None)):
            print('Dataframe has a total of ', df.shape[0], ' entries')
        else:
            print ('Dataframe has no entry, it is None')
        print('####################################################################')

    return df

def applyCut(inputDataframe, cut, text=None,verbose=False):
    dataframe = inputDataframe
    nbeforecut = dataframe.shape[0]
    cutDataframe = dataframe.query(cut)
    if text and verbose:
        print (text, cutDataframe.shape[0], ' fraction kept: %2.1f'%(100.0*float(cutDataframe.shape[0])/nbeforecut))
    return cutDataframe

def applyCutsJets(df,isMC=False,verbose=False):
    temp = df
    
    temp.eval('E_sq = gene_px**2 + gene_py**2 + gene_pz**2 + 0.00051099895**2', inplace=True)  
    temp.eval('sine_theta_e = sqrt(gene_px**2 + gene_py**2)/sqrt(gene_px**2 + gene_py**2 + gene_pz**2)', inplace=True)      
    temp.eval('gen_y_sigma = 1-E_sq*sine_theta_e**2/gen_Q2', inplace=True)      
    

    temp['pass_reco'] = np.where(temp['jet_pt']>0, 1, 0)
    if (isMC):
        temp['pass_truth'] = np.where(temp['genjet_pt']*temp['gen_Q2']>0, 1, 0)
        temp['pass_fiducial'] = np.where(temp['pass_truth']*(temp['gen_Q2'] > 150)*
                                         (temp['gen_y_sigma']>0.2)*(temp['gen_y_sigma']<0.7)*
                                         (temp['genjet_pt']>10)*
                                         (temp['genjet_eta']<2.5)*
                                         (temp['genjet_eta']>-1.), 1, 0)
        
    #temp = applyCut(temp, 'abs(vertex_z)<25 and vertex_z!=0','abs(vertex_z)<25 and and vertex_z!=0')
    #temp = applyCut(temp, 'tau1b>0 and tau1b<1', '0<tau1b<1')
    temp.eval('jet_px = jet_pt*cos(jet_phi)', inplace=True)
    temp.eval('jet_py = jet_pt*sin(jet_phi)', inplace=True)
    temp.eval('jet_pz = jet_pt*sinh(jet_eta)', inplace=True)

    temp.eval('jet_qt = sqrt( (jet_px + e_px)**2 + (jet_py + e_py)**2) ', inplace=True)
    temp.eval('jet_qtnorm = jet_qt/sqrt(Q2)', inplace=True)
    temp.eval('e_pt = sqrt(e_px*e_px + e_py*e_py)',inplace=True)

    temp.eval('e_phi1 = arctan(e_py/e_px)', inplace=True)  
    temp.eval('jet_phi1 = arctan(jet_py/jet_px)',inplace=True)
    temp.eval('jet_dphi = abs(jet_phi1 - e_phi1)',inplace=True) 
    
    temp.eval('qt_px = jet_px + e_px', inplace=True)
    temp.eval('qt_py = jet_py + e_py', inplace=True)
    temp.eval('qt_phi = arctan(qt_py/qt_px)',inplace=True)
    temp.eval('qt_dot_ept = (qt_px*e_px + qt_py*e_py)/(jet_qt*e_pt)', inplace=True)
    temp.eval('qt_dphi = arccos(qt_dot_ept)', inplace=True)
    temp.eval('qt_cos2phi = cos(2*qt_dphi)', inplace=True)
    

    
    #temp.eval('e_pt_dot_jet_pt = (e_px*jet_px + e_py*jet_py)/(e_pt*jet_pt)', inplace=True)
    #temp.eval('jet_dphi = arccos(e_pt_dot_jet_pt)',inplace=True) 
    
    temp.eval('logQ2= log(Q2)/2.3025850', inplace=True)
    temp.eval('Q = sqrt(Q2)', inplace=True)
    temp.eval('pthoverpte = pth/e_pt', inplace=True)
    temp = applyCut(temp, 'pass_reco==0 | ptmiss < 10', 'ptmiss<10',verbose)

    temp = applyCut(temp, 'pass_reco==0 | 0.08 < y < 0.7', '0.08 < y < 0.7',verbose)
    temp = applyCut(temp, 'pass_reco==0 | Q2>150', 'Q2>150',verbose)
   # temp = applyCut(temp, 'pass_reco==0 | Q2<10000', 'Q2<10000')
    temp = applyCut(temp, 'pass_reco==0 | Empz<65', 'Empz<65',verbose)
    temp = applyCut(temp, 'pass_reco==0 | Empz>45', 'Empz>45',verbose)
    temp = applyCut(temp, 'pass_reco==0 | jet_pt>5.0', 'jet pT > 5 GeV',verbose)
    temp = applyCut(temp, 'pass_reco==0 | jet_pt<150.0', 'jet pT < 150 GeV',verbose)

    temp = applyCut(temp, 'pass_reco==0 | jet_eta>-1.5', 'jet eta > -1.5',verbose)
    temp = applyCut(temp, 'pass_reco==0 | jet_eta<2.75', 'jet eta < 2.75',verbose)

    if(isMC):
        temp = applyCut(temp,'pass_truth>0',' pass_truth>0',verbose)

        temp.eval('gen_logQ2= log(gen_Q2)/2.3025850', inplace=True)   
        temp.eval('gen_Q    = sqrt(gen_Q2)', inplace=True)
        temp.eval('gene_pt = sqrt(gene_px*gene_px + gene_py*gene_py)',inplace=True)
        temp.eval('genjet_px = genjet_pt*cos(genjet_phi)', inplace=True)
        temp.eval('genjet_py = genjet_pt*sin(genjet_phi)', inplace=True)
        temp.eval('genjet_pz = genjet_pt*sinh(genjet_eta)', inplace=True)

        
        temp.eval('genjet_qt = sqrt( (genjet_px + gene_px)**2 + (genjet_py + gene_py)**2) ', inplace=True)
        temp.eval('genjet_qtnorm = genjet_qt/sqrt(gen_Q2)', inplace=True)
  
        temp.eval('gene_phi1 = arctan(gene_py/gene_px)', inplace=True)    
        temp.eval('genjet_phi1 = arctan(genjet_py/genjet_px)',inplace=True)
        temp.eval('genjet_dphi = abs(gene_phi1 - genjet_phi1)',inplace=True) 
        
     
        
        #temp.eval('gene_pt_dot_jet_pt = (gene_px*genjet_px + gene_py*genjet_py)/(gene_pt*genjet_pt)', inplace=True)
        #temp.eval('genjet_dphi = arccos(gene_pt_dot_jet_pt)',inplace=True)   
        

        temp.eval('genqt_px = genjet_px + gene_px', inplace=True)
        temp.eval('genqt_py = genjet_py + gene_py', inplace=True)
        temp.eval('genqt_phi = arctan(genqt_py/genqt_px)',inplace=True)
        temp.eval('genqt_dot_ept = (genqt_px*gene_px + genqt_py*gene_py)/(genjet_qt*gene_pt)', inplace=True)
        temp.eval('genqt_dphi = arccos(genqt_dot_ept)', inplace=True)
        temp.eval('genqt_cos2phi = cos(2*genqt_dphi)', inplace=True)

    #    temp.eval('genjet_qtnormept= genjet_qt/e_pt', inplace=True)
    #    temp.eval('genjet_qtnormjetpt= genjet_qt/genjet_pt', inplace=True)

    #Save only the features we need.
    if (isMC):
        temp = temp[['gene_px','gene_py','gene_pz','e_px','e_py','e_pz','genjet_pt',
                 'jet_pt','genjet_phi','jet_phi','genjet_eta','jet_eta','genjet_qtnorm',
                     'jet_qtnorm','genjet_dphi','jet_dphi', 'gen_y_sigma','gen_Q2','wgt','pass_reco','pass_truth', 'pass_fiducial']]
    else:
        temp = temp[['e_px','e_py','e_pz','jet_pt','jet_phi','jet_eta','jet_qtnorm','jet_dphi','wgt','pass_reco']]
        
    #df = applyCut(df, 'n_total>1', ' n>1')
    return temp


for mc_i in range(12):


    
    #Load files
    mc_name = mc_names[mc_i]
    mc_tag = mc_tags[mc_i]
    mc_tag_out = mc_tags_out[mc_i]

    print("on sample:",mc_name,mc_tag)

    mc = get_Dataframe(path, name=mc_name, tag=mc_tag, verbose=True)
    mc   = applyCutsJets(mc, isMC=True,verbose=True)
    mc.to_pickle('/clusterfs/ml4hep/yxu2/'+mc_name+"_"+mc_tag_out+".pkl")

    del mc
    gc.collect()
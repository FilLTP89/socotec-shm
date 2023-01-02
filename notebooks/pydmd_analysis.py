## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti"
__copyright__ = "Copyright 2022, CentraleSupélec (LMPS UMR CNRS 9026)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, sys
from os import listdir
from os.path import join as opj
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


import re
import pandas as pd
import numpy as np
from pydmd import DMD, MrDMD, SubspaceDMD


from read_datalogger import prep_GANTNER_dat_txt
from scipy.signal import detrend, decimate
from scipy.integrate import cumulative_trapezoid as ict


parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

pathRawData =  os.path.join(os.path.abspath(""),r'..',r'data',r'rawdat_data')
pathData = os.path.join(os.path.abspath(""),r'..',r'data',r'rawtxt_data')
pathTraitement = 'Traitement'

# datalogger = lambda s: s.split("#")[1].split("__")[0]
datehour =  lambda s: s.split("__0_")[1].split(".txt")[0]
rawfile_list = [f for f in listdir(pathData) if f.endswith('.txt')]
rawfile_list.sort(key=datehour)


# Creation du fichier vierge
df_capt_brut = pd.DataFrame(columns = ['Date_Heure','Temperature','Frequence_1_1','Frequence_2_1','Frequence_3_1','Frequence_1_2','Frequence_2_2','Frequence_3_2'])

Date_Heure = len(rawfile_list)*[0]

for i, TXT in enumerate(rawfile_list):
    date, heure = re.findall(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', TXT)[0]
    Date_Heure[i] = f'{date} {heure}'

c = len(rawfile_list)

fs = 100.0 # Hz - Sampling frequency
fmin = 0.0  # Hz
fmax = 2.0  # Hz
scaling = 1.0
r = 1

# Ajout des valeurs de fréquence au fichier

for i, TXT in enumerate(rawfile_list):
    
    # Création du dataframe du jour et de l'heure
    date, heure = re.findall(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', TXT)[0]
    df_jour = prep_GANTNER_dat_txt(pathData, TXT)
    print(TXT)
    if df_jour.empty:
        pass
    else:

        df_jour.loc[:,["Capteur_1","Capteur_2"]] = df_jour.loc[:,["Capteur_1","Capteur_2"]].apply(detrend)

        dtm = (df_jour.index[1]-df_jour.index[0]).total_seconds()
        vtm = np.array((df_jour.index-df_jour.index[0]).total_seconds())
        ntm = vtm.size
        df = 1.0/(dtm*len(vtm))
        
        fmin_samples = int(fmin/df)
        fmax_samples = int(fmax/df)
        
        t = np.linspace(start=0.0, 
                        stop=vtm[-1], 
                        num=int(vtm.size//scaling), 
                        endpoint=True)
        dt = t[1]-t[0]
        
        ddu_r_f = lambda s: np.interp(s, vtm, df_jour["Capteur_1"].values)
        ddu_θ_f = lambda s: np.interp(s, vtm, df_jour["Capteur_2"].values)

        du_r = lambda s: np.interp(s, vtm,
                                   ict(y=ddu_r_f(vtm), 
                                       x=vtm, 
                                       dx=dtm, 
                                       initial=0)
                                   )
        du_θ = lambda s: np.interp(s, vtm,
                                   ict(y=ddu_θ_f(vtm), 
                                       x=vtm, 
                                       dx=dtm, 
                                       initial=0)
                                   )
        X_r = np.nan_to_num(np.array(du_r(t), dtype=np.float64).flatten())
        X_θ = np.nan_to_num(np.array(du_θ(t), dtype=np.float64).flatten())

        # sub_dmd = SubspaceDMD(svd_rank=-1, opt=False,
        #                       rescale_mode=None, sorted_eigs=False)
        sub_dmd = DMD(svd_rank=-1)
        dmd_r = MrDMD(sub_dmd, 
                      max_level=10, 
                      max_cycles=15)
        dmd_r.fit(X=X_r.T)
        
        dmd_θ = MrDMD(sub_dmd, 
                      max_level=10, 
                      max_cycles=15)
        dmd_θ.fit(X=X_θ.T)
        
        # import pdb
        # pdb.set_trace()
        
        fig, ax = plt.subplots(2, 1, 
                               sharex=True,
                               sharey=True)

        l00 = ax[0].plot(vtm, du_r(vtm),
                         color='black',
                         label=r'$\dot{u}_r(t)$')
        l01 = ax[0].plot(vtm, dmd_r.reconstructed_data.T.real, 
                         color='red',
                         linewidth=0.6,
                         label=r'$\tilde{\dot{u}}_r(t)$')
        # l01 = ax[0].plot(vtm, dmd_r.partial_reconstructed_data(level=2).real.T,
        #                  color='red',
        #                  linewidth=0.6,
        #                  label=r'$\tilde{\dot{u}}_r(t)$')
        
        
        l10 = ax[1].plot(vtm, du_θ(vtm),
                         color='black',
                         label=r'$\dot{u}_\theta(t)$')
        l11 = ax[1].plot(vtm, dmd_θ.reconstructed_data.T.real,
                         color='red',
                         linewidth=0.6,
                         label=r'$\tilde{\dot{u}}_\theta(t)$')
        # l11 = ax[1].plot(vtm, dmd_θ.partial_reconstructed_data(level=2).real.T,
        #                  color='red',
        #                  linewidth=0.6,
        #                  label=r'$\tilde{\dot{u}}_\theta(t)$')
        
        ax[0].axis('tight')
        ax[0].set(xlim=(0.0,900.0),
                  xlabel=r'$t$ [s]', 
                  ylabel=r'$\dot{u}_r(t)$ [m/s]',
                  title='Datalogger recorded on {} at {}'.format(date, heure))
        ax[1].axis('tight')
        ax[1].set(xlim=(0.0,900.0),
                  xlabel=r'$t$ [s]', 
                  ylabel=r'$\dot{u}_\theta(t)$ [m/s]')
                #   title='Datalogger recorded on {} at {}'.format(date,heure))
        
        ax[0].legend(frameon=False)
        ax[1].legend(frameon=False)
        
        fig.savefig("{:s}_dmd.png".format(TXT.strip(".dat.txt")),
                    format="png", 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()

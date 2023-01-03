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

"""Credits: http://www.pyrunner.com/weblog/2016/07/25/dmd-python/"""
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, sys
from os import listdir
from os.path import isfile, join
import pdb
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


import re

# Time manipulation packages
import datetime
import time

## Imports 
import pandas as pd
import numpy as np
import dmd_functions as dmd

from read_datalogger import prep_GANTNER_dat_txt
from scipy.signal import detrend, decimate



parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

pathRawData =  os.path.join(os.path.abspath(""),r'..',r'data',r'rawdat_data')
pathData = os.path.join(os.path.abspath(""),r'..',r'data',r'rawtxt_data')
pathTraitement = 'Traitement'


onlyTXT = [f for f in listdir(pathData) if f.endswith('.txt')]

# Creation du fichier vierge
df_capt_brut = pd.DataFrame(columns = ['Date_Heure','Temperature','Frequence_1_1','Frequence_2_1','Frequence_3_1','Frequence_1_2','Frequence_2_2','Frequence_3_2'])

Date_Heure = len(onlyTXT)*[0]

for i, TXT in enumerate(onlyTXT):
    date, heure = re.findall(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', TXT)[0]
    Date_Heure[i] = f'{date} {heure}'

c = len(onlyTXT)

fs = 100.0 # Hz - Sampling frequency
fmin = 0.0  # Hz
fmax = 2.0  # Hz
scaling = 10.0
r = 1

# Ajout des valeurs de fréquence au fichier
for i, TXT in enumerate(onlyTXT):
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
        df = 1.0/(dtm*len(vtm))  # sampling step in frequency domain (Hz)
        
        fmin_samples = int(fmin/df)
        fmax_samples = int(fmax/df)
        extent = (vtm[0], vtm[-1], fmin, fmax)
        c1 = df_jour["Capteur_1"].values
        c2 = df_jour["Capteur_2"].values
        c1f = lambda s: np.interp(s,vtm,c1)
        c2f = lambda s: np.interp(s,vtm,c2)
        dt = dtm/scaling
        t = np.linspace(vtm[0], vtm[-1],
                        int(len(vtm)/scaling), endpoint=True)
                
        R = np.array([c1f(t),c2f(t)], dtype=np.float64)
        X = R[:,:-1]
        Y = R[:,1:]
        
        dmd.check_linear_consistency(X, Y)
        
        mu,Phi = dmd.dmd(X, Y, r)
        dmd.check_dmd_result(X, Y, mu, Phi)
        Psi, Dtil = dmd.modal_time_evolution(mu, Phi, dt, 
                                             t, X[:, 0], r) 
        
        fig, ax = plt.subplots(2, int(r), 
                               sharex=True,
                               sharey=True)

        ax[0].plot(t, np.abs(Psi.flatten()), 
                    color='black', 
                    label=r"$\vert \Psi_{{:d}} \vert$".format(r))
        ax[1].plot(t, np.angle(Psi.flatten()),
                    color='red', 
                    label=r"$\theta(\Psi_{{:d}})$".format(r))
        ax[0].axis('tight')
        ax[0].set(xlabel=r'$t$ [s]')
        ax[1].axis('tight')
        ax[1].set(xlabel=r'$t$ [s]')
        fig.savefig("{:s}_dmd.png".format(TXT.strip(".dat.txt")),
                    format="png", 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()

## -*- coding: utf-8 -*-
#!/usr/bin/env python3
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
import pywt
from stockwell import st

from read_datalogger import prep_GANTNER_dat_txt
from scipy.signal import detrend, decimate
# from scipy.fft import fft, fftfreq, fftshift
# from scipy.signal import find_peaks
# from scipy.signal.windows import get_window

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
# stock = st.st(w, fmin_samples, fmax_samples)


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
        df = 1.0/(dtm*len(vtm))  # sampling step in frequency domain (Hz)
        
        fmin_samples = int(fmin/df)
        fmax_samples = int(fmax/df)
        extent = (vtm[0], vtm[-1], fmin, fmax)
        # pdb.set_trace()
        c1 = df_jour["Capteur_1"].values
        c2 = df_jour["Capteur_2"].values
        c1f = lambda s: np.interp(s,vtm,c1)
        c2f = lambda s: np.interp(s,vtm,c2)
        t = decimate(x=vtm, q=int(scaling))
        
        # t = np.linspace(0, vtm[-1], len(vtm)//int(scaling), endpoint=False)
        # pdb.set_trace()
        st1 = st.st(c1, 
                    fmin_samples, 
                    fmax_samples, 
                    10.0, 
                    'gauss')
        st2 = st.st(c2, 
                    fmin_samples, 
                    fmax_samples, 
                    10.0, 
                    'gauss')

        # fp,tp = np.unravel_index(np.argmax(stock),stock.shape)
        
        # vtm_red = vtm[tp-1:np.argwhere(vtm==vtm[tp]+10.0)[0][0]+1]
        
        fig, ax = plt.subplots(2, 1, sharex=True)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('bottom', size='10%', pad=0.45)
        ax[0].plot(vtm, c1, color='black')
        ax[0].set(ylabel=r'$\ddot{u}_r(t)$',
                  ylim=[-0.025,0.025])
        im=ax[1].imshow(np.abs(st1),
                     origin='lower', 
                     extent=extent,
                     cmap='bone',
                     norm='log',
                     vmin=1e-5,
                     vmax=1e-3)
        ax[1].axis('tight')
        ax[1].set(xlabel=r'$t$ [s]', 
                  ylabel=r'$f$ [Hz]')  # ,
                #   yscale='log')
        fig.colorbar(im, cax=cax, orientation='horizontal')

        fig.savefig("{:s}_st1.png".format(TXT.strip(".dat.txt")),
            format="png", bbox_inches='tight',dpi = 300)
        plt.close()


        fig, ax = plt.subplots(2, 1, sharex=True)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('bottom', size='10%', pad=0.45)
        ax[0].plot(vtm, c2, color='black')
        ax[0].set(ylabel=r'$\ddot{u}_\theta(t)$',
                  ylim=[-0.025,0.025])
        ax[1].imshow(np.abs(st2),
                     origin='lower', 
                     extent=extent,
                     cmap='bone',
                     norm='log',
                     vmin=1e-5,
                     vmax=1e-3)
        ax[1].axis('tight')
        ax[1].set(xlabel=r'$t$ [s]', 
                  ylabel=r'$f$ [Hz]')#,
                #   yscale='log')
        fig.colorbar(im, cax=cax, orientation='horizontal')
        fig.savefig("{:s}_st2.png".format(TXT.strip(".dat.txt")),
            format="png", bbox_inches='tight',dpi = 300)
        plt.close()

## -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import sys
from os import listdir
from os.path import join as opj
import dask.dataframe as dd
from matplotlib import pyplot as plt
import re
import pandas as pd
import numpy as np
from pydmd import SubspaceDMD
from scipy.signal import detrend
from scipy.integrate import cumulative_trapezoid
import pdb
u"""General informations"""
__author__ = "Filippo Gatti"
__copyright__ = "Copyright 2022, CentraleSupélec (LMPS UMR CNRS 9026)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Path2RawData = os.path.join(os.path.abspath(""),r'..',r'data',r'rawtxt_data')

# datalogger = lambda s: s.split("#")[1].split("__")[0]
# datehour =  lambda s: s.split("__0_")[1].split(".filename")[0]
# rawfile_list = [f for f in listdir(Path2RawData) if f.endswith('.filename')]
# rawfile_list.sort(key=datehour)

# Path2CsvData = os.path.join(os.path.abspath(""), r'..', r'data', r'csv_data')
# csvfile_list = [f for f in listdir(Path2CsvData) if f.endswith('.csv')]
# csvfile_list.sort(key=datehour)


Path2ParquetData = os.path.join(os.path.abspath(""),
                                r'..', r'data', r'parquet_data')

# Date_Heure = len(rawfile_list)*[0]
# for i, filename in enumerate(rawfile_list):
#     date, heure = re.findall(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', filename)[0]
#     Date_Heure[i] = f'{date} {heure}'

fs = 100.0 # Hz - Sampling frequency
fmin = 0.0  # Hz
fmax = 2.0  # Hz
scaling = 1.0
r = 1



df = dd.read_parquet(os.path.join(Path2ParquetData, "Datalogger.parquet"),
                     engine='pyarrow')
df = df.set_index("Date_Heure", sorted=True)

df = df.map_partitions(lambda df: df.assign(ü_r=detrend(df.Capteur_1), 
                                            ü_θ=detrend(df.Capteur_2)), 
                       meta={"Capteur_1": 'f8', 
                             "Capteur_2": 'f8', 
                             "Temperature": 'f8', 
                             "ü_r": 'f8', 
                             "ü_θ": 'f8'}
                       ).drop("Capteur_1", axis=1).drop("Capteur_2", axis=1)
df = df.map_partitions(lambda df: df.assign(ũ_r=cumulative_trapezoid(y=df.ü_r,
                                                                     dx=0.01,
                                                                     initial=0),
                                            ũ_θ=cumulative_trapezoid(y=df.ü_θ,
                                                                     dx=0.01,
                                                                     initial=0)),
                       meta={"Temperature": 'f8',
                             "ü_r": 'f8',
                             "ü_θ": 'f8',
                             "ũ_r": 'f8',
                             "ũ_θ": 'f8',}
                       )

# sub_dmd = DMD(svd_rank=-1, forward_backward=True)
# dmd_r = MrDMD(sub_dmd,
#               max_level=10,
#               max_cycles=15)
# dmd_θ = MrDMD(sub_dmd,
#               max_level=10,
#               max_cycles=15)
dmd = SubspaceDMD(svd_rank=3, opt=True,)


# vtm = df.loc[:"2022", :].index.compute()
# vtm_dmd = vtm.to_numpy()[np.argwhere(np.logical_or(vtm.year == 2016, vtm.year == 2017))]
dmd.fit(X=df.loc["2016-10-26", 
                 ["ũ_r", 
                  "ũ_θ",]
                 ].sample(frac=0.1).compute().values.reshape(-1, 2).T)
# dmd_θ.fit(X=df.loc["2016":"2017", "ũ_θ"].compute().values.reshape(1, -1))

# dmd_r.dmd_time['dt']=
dmd.dmd_time['tend'] *= 3.0

pdb.set_trace()

fig, ax = plt.subplots(3, 1,
                       sharex=True,
                       sharey=False,
                       figsize=(6, 3.5))

average = "1s"
df.loc["2016-10-26":"2016-10-27",["ũ_r"]].compute().plot(
    ax=ax[0], color='black', legend=False, )
df.loc["2016-10-26":"2016-10-27",["ũ_θ"]].compute().plot(
    ax=ax[1], color='black', legend=False, )
df.loc["2016-10-26":"2016-10-27",["Temperature"]].compute().plot(
    ax=ax[2], color='black', legend=False, )


# l01 = ax[0].plot(vtm, dmd_r.reconstructed_data.T.real,
l01 = ax[0].plot(vtm, dmd_r.reconstructed_data.T.real,
                 color='red',
                 linewidth=0.6,
                 label=r'$\tilde{\dot{u}}_r(t)$')
# l01 = ax[0].plot(vtm, dmd_r.partial_reconstructed_data(level=2).real.T,
#                  color='red',
#                  linewidth=0.6,
#                  label=r'$\tilde{\dot{u}}_r(t)$')

l11 = ax[1].plot(vtm, dmd_θ.reconstructed_data.T.real,
                 color='red',
                 linewidth=0.6,
                 label=r'$\tilde{\dot{u}}_\theta(t)$')
# l11 = ax[1].plot(vtm, dmd_θ.partial_reconstructed_data(level=2).real.T,
#                  color='red',
#                  linewidth=0.6,
#                  label=r'$\tilde{\dot{u}}_\theta(t)$')

ax[0].axis('tight')
ax[0].set(xlim=(0.0, 900.0),
          xlabel=r'$t$ [s]',
          ylabel=r'$\dot{u}_r(t)$ [m/s]',
          title='Datalogger recorded on {} at {}'.format(date, heure))
ax[1].axis('tight')
ax[1].set(xlim=(0.0, 900.0),
          xlabel=r'$t$ [s]',
          ylabel=r'$\dot{u}_\theta(t)$ [m/s]')
#   title='Datalogger recorded on {} at {}'.format(date,heure))

ax[0].legend(frameon=False)
ax[1].legend(frameon=False)

fig.savefig("dmd.png",
            format="png",
            bbox_inches='tight',
            dpi=300)
plt.close()

# Ajout des valeurs de fréquence au fichier

# for i, filename in enumerate(csvfile_list):
    
#     # Création du dataframe du jour et de l'heure
#     date, heure = re.findall(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', filename)[0]
#     # df_jour = prep_GANTNER_dat_filename(Path2RawData, filename)
#     # print(filename)
#     # if df_jour.empty:
#     #     pass
#     # else:
    
#         df_jour.loc[:,["Capteur_1","Capteur_2"]] = df_jour.loc[:,["Capteur_1","Capteur_2"]].apply(detrend)

#         dtm = (df_jour.index[1]-df_jour.index[0]).total_seconds()
#         vtm = np.array((df_jour.index-df_jour.index[0]).total_seconds())
#         ntm = vtm.size
#         df = 1.0/(dtm*len(vtm))
        
#         fmin_samples = int(fmin/df)
#         fmax_samples = int(fmax/df)
        
#         t = np.linspace(start=0.0, 
#                         stop=vtm[-1], 
#                         num=int(vtm.size//scaling), 
#                         endpoint=True)
#         dt = t[1]-t[0]
        
#         ddu_r_f = lambda s: np.interp(s, vtm, df_jour["Capteur_1"].values)
#         ddu_θ_f = lambda s: np.interp(s, vtm, df_jour["Capteur_2"].values)

#         du_r = lambda s: np.interp(s, vtm,
#                                    ict(y=ddu_r_f(vtm), 
#                                        x=vtm, 
#                                        dx=dtm, 
#                                        initial=0)
#                                    )
#         du_θ = lambda s: np.interp(s, vtm,
#                                    ict(y=ddu_θ_f(vtm), 
#                                        x=vtm, 
#                                        dx=dtm, 
#                                        initial=0)
#                                    )
#         X_r = np.nan_to_num(np.array(du_r(t), dtype=np.float64).flatten())
#         X_θ = np.nan_to_num(np.array(du_θ(t), dtype=np.float64).flatten())

#         # sub_dmd = SubspaceDMD(svd_rank=-1, opt=False,
#         #                       rescale_mode=None, sorted_eigs=False)
#         
        
#         # import pdb
#         # pdb.set_trace()
        
        

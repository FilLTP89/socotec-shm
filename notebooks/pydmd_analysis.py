## -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import sys
from os import listdir
from os.path import join as opj
import xarray as xr
import dask.dataframe as dd
from matplotlib import pyplot as plt
import re
import pandas as pd
import numpy as np
from ezyrb import POD, RBF
from pydmd import HODMD, ParametricDMD
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

Path2ParquetData = os.path.join(os.path.abspath(""),
                                r'..', r'data', r'parquet_data')

df = dd.read_parquet(os.path.join(Path2ParquetData, "Datalogger.parquet"),
                     engine='pyarrow')
df = df.set_index("Date_Heure", sorted=True)

df = df.map_partitions(lambda df: df.assign(a_r=detrend(df.Capteur_1), 
                                            a_t=detrend(df.Capteur_2)), 
                       meta={"Capteur_1": 'f8', 
                             "Capteur_2": 'f8', 
                             "Temperature": 'f8', 
                             "a_r": 'f8', 
                             "a_t": 'f8'}
                       ).drop("Capteur_1", axis=1).drop("Capteur_2", axis=1)
df = df.map_partitions(lambda df: df.assign(v_r=cumulative_trapezoid(y=df.a_r,
                                                                     dx=0.01,
                                                                     initial=0),
                                            v_t=cumulative_trapezoid(y=df.a_t,
                                                                     dx=0.01,
                                                                     initial=0)),
                       meta={"Temperature": 'f8',
                             "a_r": 'f8',
                             "a_t": 'f8',
                             "v_r": 'f8',
                             "v_t": 'f8',}
                       )

fss = '10S'
df = df.resample(fss).first()

# T = df.Temperature.map_partitions(lambda x: x.mean()).compute()


# X_r = df.v_r.sort_values("Date_Heure").resample(fss).first().compute()
# X_t = df.v_t.sort_values("Date_Heure").resample(fss).first().compute()

# list_of_delayed = df.map_partitions(lambda x: x.v_r).to_delayed().to_list()
# X_r = dask.compute(*list_of_delayed)

# xa = xr.DataArray(df[["v_r", "v_t", "Temperature"]],
#                   coords=[df.index, ["v_r", "v_t", "T"]],
#                   dims=("Time", "Datalog")).sortby("Time")
chunk_sizes = df.map_partitions(len).compute().values
da = df[["v_r", "v_t", "Temperature"]].to_dask_array(lengths=chunk_sizes)
pdb.set_trace()
#https: // stackoverflow.com/questions/72015205/iterating-through-dask-array-chunks

dmds = [HODMD(svd_rank=0,
              forward_backward=True,
              exact=True,
              opt=True,
              d=100)]
pdmd = ParametricDMD(dmds,
                     POD(svd_rank=-1),
                     RBF())



#######
# X_r = df.loc["2016-10-26 17:11:40":"2016-10-26 17:26:39",
#              ["v_r"]
#              ].sort_values("Date_Heure").resample(fss).first().compute()
# X_t = df.loc["2016-10-26 17:11:40":"2016-10-26 17:26:39",
#              ["v_t"]
#              ].sort_values("Date_Heure").resample(fss).first().compute()
# dmd = HODMD(svd_rank=0,
#             forward_backward=True,
#             exact=True,
#             opt=True,
#             d=100)

# dmd.fit(X=X_r.values.reshape(-1, 1).T)


# dmd.dmd_time['tend'] *= 2 # int((24*60+15)*60*100/int(fss.split('S')[0]))

# dtm = (X_r.index.values[1] - X_r.index.values[0])/np.timedelta64(1, 's')
# vt0 = X_r.index.values[0]

# dmd_vtm = vt0 + (dmd.dmd_timesteps*dtm*1e9).astype('timedelta64[ns]')
########

# plt.plot(X_r.values.reshape(-1, 2))
# plt.plot(dmd.reconstructed_data.real.T)
# 
# plt.show()

# 

fig, ax = plt.subplots(1, 1,
                       sharex=True,
                       sharey=True,
                       figsize=(6, 3.5))







fig, ax = plt.subplots(3, 1,
                       sharex=True,
                       sharey=False,
                       figsize=(6, 3.5))

# average = "1s"
df.loc["2016-10-26":"2016-10-27",["v_r"]].compute().plot(
    ax=ax[0], color='black', legend=r"recorded", )
df.loc["2016-10-26":"2016-10-27",["v_t"]].compute().plot(
    ax=ax[1], color='black', legend=r"recorded", )
df.loc["2016-10-26":"2016-10-27",["Temperature"]].compute().plot(
    ax=ax[2], color='black', legend=r"recorded", )


l01 = ax[0].plot(dmd_vtm, dmd.reconstructed_data.real[0,:],
                 color='red',
                 linewidth=2,
                 label=r'DMD')


# l11 = ax[1].plot(dmd_vtm, dmd.reconstructed_data.real[1, :],
#                  color='red',
#                  linewidth=2,
#                  label=r'DMD')

ax[0].axis('tight')
ax[0].set(xlabel=r'$t$ [s]',
          ylabel=r'$\dot{u}_r(t)$ [m/s]',
          title='Datalogger')
ax[1].axis('tight')
ax[1].set(xlabel=r'$t$ [s]',
          ylabel=r'$\dot{u}_\theta(t)$ [m/s]')

ax[0].legend(frameon=False)
ax[1].legend(frameon=False)


# fig.savefig("dmd.png",
#             format="png",
#             bbox_inches='tight',
#             dpi=300)
plt.show()

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
#         ddu_t_f = lambda s: np.interp(s, vtm, df_jour["Capteur_2"].values)

#         du_r = lambda s: np.interp(s, vtm,
#                                    ict(y=ddu_r_f(vtm), 
#                                        x=vtm, 
#                                        dx=dtm, 
#                                        initial=0)
#                                    )
#         du_t = lambda s: np.interp(s, vtm,
#                                    ict(y=ddu_t_f(vtm), 
#                                        x=vtm, 
#                                        dx=dtm, 
#                                        initial=0)
#                                    )
#         X_r = np.nan_to_num(np.array(du_r(t), dtype=np.float64).flatten())
#         X_t = np.nan_to_num(np.array(du_t(t), dtype=np.float64).flatten())

#         # sub_dmd = SubspaceDMD(svd_rank=-1, opt=False,
#         #                       rescale_mode=None, sorted_eigs=False)
#         
        
#         # import pdb
#         # pdb.set_trace()
        
        

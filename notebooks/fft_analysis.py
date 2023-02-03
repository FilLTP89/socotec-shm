## -*- coding: utf-8 -*-
#!/usr/bin/env python3

from fast_konno_ohmachi import fast_konno_ohmachi as fko
import os
import sys
import math
import dask.dataframe as dd
# import dask.bag as db
import dask.array as da
import dask.array.fft as dfft
from dask.distributed import Client
from matplotlib import pyplot as plt
import pdb
import numpy as np
from scipy.signal import detrend
from scipy.integrate import cumulative_trapezoid

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

os.environ['DASK_SCHEDULER_ADDRESS'] = 'tcp://localhost:8786'
# Client()
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def next_power_of_2(x):
    return 2**1 if x == 0 else 2**math.ceil(math.log2(x))

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

fss = '0.25S'
dtm = float(fss.strip("S"))
# rec = 15*60*np.timedelta64(fss.strip('S'), 's').astype(np.int64)
df = df.resample(fss).first().dropna()

chunk_sizes = list(df.map_partitions(len).compute().values)

ar = df[["v_r", "v_t", "Temperature"]].to_dask_array(lengths=chunk_sizes)

nfq = np.stack([next_power_of_2(b.size) 
                for b in ar[:, 2].blocks])
vfq = [np.linspace(0.0,
                   0.5/dtm,
                   n//2+1,
                   endpoint=True) 
       for n in nfq]

FSS = ar.map_blocks(lambda ths: np.fft.rfft(ths, 
                                            n=next_power_of_2(ths.shape[0]), 
                                            axis=0),
                    enforce_ndim=True,
                    dtype=np.complex128, 
                    chunks=(tuple([next_power_of_2(ths.shape[0])//2+1 for ths in ar.blocks]), 
                            (3,)
                            ),
                    meta=np.array((),
                                  dtype=np.complex128),
                    )

TFS = FSS.map_blocks(lambda fss: np.stack([fko(np.abs(FSS.blocks[0][1:, 0].T/
                                                      FSS.blocks[0][1:, 2].T), 
                                               np.linspace(0.0, 0.5/dtm, 
                                                           FSS.blocks[0].shape[0], 
                                                           endpoint=True)[1:], 
                                               smooth_coeff=40), 
                                           fko(np.abs(FSS.blocks[0][1:, 1].T/
                                                      FSS.blocks[0][1:, 2].T), 
                                               np.linspace(0.0, 0.5/dtm, 
                                                           FSS.blocks[0].shape[0], 
                                                           endpoint=True)[1:], 
                                               smooth_coeff=40)]).T,
                     dtype=np.float64,
                     chunks=(tuple([fs.shape[0]-1 for fs in FSS.blocks]),
                             (2,)
                             ),
                     meta=np.array((),
                                   dtype=np.float64),
                     )

pdb.set_trace()


plt.loglog(vfq[0], np.abs(FSS.blocks[0][:, 0]/FSS.blocks[0][:, 2]), color='black')
plt.loglog(vfq[0][1:], TFS.blocks[0][:, 0], color='orange')
plt.show()
pdb.set_trace()
for i,f in enumerate(vfq):
    plt.semilogx(f, np.abs(FSS[i][:, 0]/FSS[i][:, 1]), color='black')

pdb.set_trace()

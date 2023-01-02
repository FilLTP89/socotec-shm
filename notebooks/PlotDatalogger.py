## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
import os
from scipy.signal import detrend
import dask.dataframe as dd
import xarray as xr

from matplotlib import pyplot as plt

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

# client = Client(n_workers=2, 
#                 threads_per_worker=2, 
#                 memory_limit="1GB")

pathData = os.path.join(os.path.abspath(""),
                        r'..',r'data',r'parquet_data')
df = dd.read_parquet(os.path.join(pathData, "Datalogger.parquet"), 
                     engine='pyarrow')
df = df.set_index("Date_Heure", sorted=True)
df = df.persist()
da = xr.DataArray(df.to_dask_array(lengths=True),
                  dims=("Time", "Datalog"))


def detrend_dim(da, dim, deg=1):
    """detrend along a single dimension."""
    # calculate polynomial coefficients
    p = da.polyfit(dim=dim, deg=deg, skipna=False)
    # evaluate trend
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    # remove the trend
    return da - fit

detrended = detrend_dim(da, dim="Time")


fig, ax = plt.subplots(3, 1,
                       sharex=True,
                       sharey=False,
                       figsize=(6, 3.5))



average = "24h"
df[["Capteur_1"]].resample(average).mean().compute().plot(
    ax=ax[0], color='black', legend=False, )
df[["Capteur_2"]].resample(average).mean().compute().plot(
    ax=ax[1], color='black', legend=False, )
df[["Temperature"]].resample(average).mean().compute().plot(
    ax=ax[2], color='black', legend=False, )


# plt.show()

# ax[0].get_legend().remove() #.legend(labels=(r'$\ddot{u}_r(t)$'))

# ax[1].get_legend().remove() #.legend(labels=(r'$\ddot{u}_\theta(t)$'))
# ax[2].get_legend().remove() #.legend(labels=(r'$T(t)$'))

ax[0].axis('tight')
ax[0].set(xlabel=r'$t$ [months]', 
          ylabel=r'$\ddot{u}_r(t)$ [m/s/s]',
          ylim=(-0.04,0.04))
ax[1].axis('tight')
ax[1].set(xlabel=r'$t$ [months]', 
          ylabel=r'$\ddot{u}_\theta(t)$ [m/s/s]',
          ylim=(-0.04,0.04))

ax[2].axis('tight')
ax[2].set(xlabel=r'$t$ [months]', 
          ylabel=r'$T(t)$ [$^\circ$ C]')

# ax[0].legend(frameon=False)
# ax[1].legend(frameon=False)
# ax[2].legend(frameon=False)
fig.savefig("datalogger_{:>s}.png".format(average),
            format="png", 
            bbox_inches='tight', 
            dpi=300)
plt.close()

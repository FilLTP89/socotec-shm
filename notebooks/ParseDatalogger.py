## -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import codecs
import pandas as pd
import dask as dk
import dask.dataframe as dd
import re


def datehour(s): return s.split("__0_")[1].split(".txt")[0]

def Txt2Parquet(pathData=None):
    Txt2Csv(pathData)
    Csv2Parquet(pathData)
    
def Csv2Parquet(pathData=None):
    if not pathData:
        pathData = os.path.join(os.path.abspath(
            ""), r'..', r'data', r'csv_data')
    rawfile_list = [f for f in os.listdir(pathData) if f.endswith('.txt')]
    rawfile_list.sort(key=datehour)
    datalog = dd.read_csv(os.path.join(pathData,
                                       "{:>s}".format("Datalogger_*.csv")),
                          parse_dates=["Date_Heure"])
    datalog.to_parquet(os.path.join(pathData.replace("csv","parquet"),
                                    "Datalogger.parquet"),
                       engine='pyarrow')


def Txt2Csv(pathData=None, correct=False):
    if not pathData:
        pathData=os.path.join(os.path.abspath(""), r'..', r'data', r'rawtxt_data')
    rawfile_list = [f for f in os.listdir(pathData) if f.endswith('.txt')]
    rawfile_list.sort(key=datehour)
    
    for i, TXT in enumerate(rawfile_list):

        # Création du dataframe du jour et de l'heure
        date, heure = re.findall(
            r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', TXT)[0]
        try:
            df = prep_GANTNER_dat_txt(pathData, TXT)
        except:
            pass
        print(TXT)
        if df.empty:
            pass
        else:
            df.to_csv(os.path.join(pathData.replace("rawtxt","csv"),
                                   "{:>s}.csv".format(TXT.split(".dat.txt")[0])))
    return

def prep_GANTNER_dat_txt(data_path, file):

    # Extracting date and time
    date, heure = re.findall(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', file)[0]

    file_path = os.path.join(data_path,file)

    # File as df
    df_jour = read_GANTNER_dat_txt(file_path)

    # Creating time indexes
    df_jour.loc[:,'Date_Heure'] = pd.date_range(
        start = pd.to_datetime(
            f'{date} {heure}',
            format = '%Y-%m-%d %H-%M-%S'),
        freq ='0.01S', 
        periods = len(df_jour)
        )

    # Reseting indexes
    df_jour.index = df_jour['Date_Heure']
    try:
        df_jour = df_jour[['accelero1(x)','accelero2(y)','PT100']].rename(
            columns = {'accelero1(x)':'Capteur_1','accelero2(y)':'Capteur_2',
                       'PT100' : 'Temperature'})
        df_jour = df_jour.apply(lambda x: x.str.replace(',','.')).astype(float)
        return df_jour
    except:
        return pd.DataFrame()

def read_GANTNER_dat_txt(fileName, export_dict = False):
    '''
    Lecture d'un fichier .dat.txt type VDV décodé sous la forme d'un dictionnaire contenant toutes les informations et dataFrame pandas.

    Parameters
    ----------
    fileName : str
        la localisation du fichier dat.txt à lire
    export_dict : bool, optional
        Variable permettant d'exporter ou non le dictionnaire d'information préliminaire du fichier. The default is False.

    Returns
    -------
    df_file : pd.DataFrame
        dataframe des données du dat.txt
    dict_file : dict
        dictionnaire des informations du dat.txt

    Examples
    -------

    '''
    dict_file = {}
    with codecs.open(fileName, 'r', 'windows-1252') as my_file: #'utf8', 
        for (n,line) in enumerate(my_file):
            if line.encode('utf-8').strip():
                if re.search(r'[*]+', line):
                    break
                else:
                    info_match = re.findall(r'([A-Za-z]*): ([^\/\n]*)', line) 

                    for key,value in info_match:
                        if key in dict_file:
                            dict_file[key].append(value.replace(" ", ""))
                        else:
                            dict_file.update({key: [value.replace(" ", "")]})
        df_file = pd.read_csv(fileName,
                              skiprows = range(n+1),
                              sep = '\t',
                              usecols = list(range(1,len(dict_file['Name'])+1)),
                              names = dict_file['Name'],
                              encoding='unicode_escape')
        df_file = (df_file)

    if export_dict == True:
        return df_file, dict_file
    else:
        return df_file
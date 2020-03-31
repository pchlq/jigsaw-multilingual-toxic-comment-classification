import pandas as pd
import argparse
from googletrans import Translator
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import random
import time
import json
import subprocess


listofservers = ["South Africa", "Egypt" , "Australia", "New Zealand",  "South Korea", 
        "Singapore", "Taiwan", "Vietnam", "Hong Kong", "Indonesia", "Thailand", "Japan", "Malaysia", 
        "United Kingdom", "Netherlands", "Germany", "France", "Belgium", "Switzerland", "Sweden","Spain",
        "Denmark", "Italy", "Norway", "Austria", "Romania", "Czech Republic", "Luxembourg", "Poland", 
        "Finland", "Hungary", "Latvia", "Russia", "Iceland", "Bulgaria", "Croatia", "Moldova", "Portugal", 
        "Albania", "Ireland", "Slovakia","Ukraine", "Cyprus", "Estonia", "Georgia", "Greece", "Serbia", "Slovenia", 
        "Azerbaijan", "Bosnia and Herzegovina", "Macedonia","India", 'Turkey', 'Israel', 'United Arab Emirates', 
        'United States', 'Canada','Mexico',"Brazil", "Costa Rica", "Argentina", "Chile"
        ]

def SelectServer(l):
    return random.choice(l)


def translate_item( idx: 'int', 
                    row: 'str', 
                    lg: 'str'
                    ) -> tuple:

    """
    make sure that NordVPN is installed and running
    """
    
    if lg !='en':
        translator = Translator()
        try:
            result = translator.translate(row, src=lg)
            return idx, result.text
        except json.decoder.JSONDecodeError:
            print("exception !! VPN disconnection")
            process = subprocess.Popen(["nordvpn", "-d"], shell = True ,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()

            time.sleep(5)

            srv = SelectServer(listofservers)

            print("server selection  : "+ srv + " and connection")

            process = subprocess.Popen(["nordvpn", "-c", "-g", srv ], shell = True ,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
            time.sleep(100)

            return translate_item(idx, row, lg)
    else:
        return idx, row


if __name__=='__main__':
    """
    Example of usage:
    >>> python translate_sentense.py --path /input --fname test.csv
    """

    parser = argparse.ArgumentParser(description="Translate sentense")
    parser.add_argument("--path", help="path to file", type=str, default='input')
    parser.add_argument("--path_to_save", help="path to save", type=str, default='test_fold')
    parser.add_argument("--fname", help="file name", type=str, default='test.csv')
    parser.add_argument("--n_jobs", help="numbers of cores to job", type=int, default=8)

    args = parser.parse_args()
    path = args.path
    path_to_save = args.path_to_save
    fname = args.fname
    n_jobs = args.n_jobs

    df = pd.read_csv(f"{path}/{fname}")
    name = fname.split('.')[0]
    
    for chunk in tqdm( range(len(df) // 301) ):
        df_tmp = df.iloc[chunk*301 : (chunk+1)*301, :].copy()
        d={}
        d.update(Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(translate_item)(idx, row, lg) for idx, (row, lg )in tqdm(enumerate(zip(df_tmp.content.values, 
                                                                                           df_tmp.lang.values)))
        ))

        df_tmp["content_en"] = df_tmp.id.map(d)
        df_tmp["content_en"].fillna("none", inplace=True)
        df_tmp.to_csv(f"{path_to_save}/{name}_en_{chunk}.csv")
        print(f"Chunk {chunk} saved successfuly")

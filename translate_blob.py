import pandas as pd
import argparse
from textblob import TextBlob, exceptions
from tqdm import tqdm
import time


def translate_blob(row: 'str', 
                   lang: 'str'
                  ) -> tuple:
    
    try:
        blob = TextBlob(row)
        result = str(blob.translate(from_lang=lang, to='en'))
        time.sleep(0.2)
    except exceptions.NotTranslated:
        print('pass..')
        result = 'pass'
      
    return result


if __name__=='__main__':
    """
    Example of usage:
    >>> python translate_blob.py --path input --fname test.csv

    make sure that VPN is installed and running
    """

    parser = argparse.ArgumentParser(description="Translate sentense")
    parser.add_argument("--path", help="path to file", type=str, default='input')
    parser.add_argument("--path_to_save", help="path to save", type=str, default='test_fold')
    parser.add_argument("--fname", help="file name", type=str, default='test.csv',
                                             choices=['test.csv', 'validation.csv'])

    args = parser.parse_args()
    path = args.path
    path_to_save = args.path_to_save
    fname = args.fname

    df = pd.read_csv(f"{path}/{fname}")
    df_tmp = df.copy()
    name = fname.split('.')[0]

    if name == 'test':
        chunk_size = 301 # test (301 x 212)
        text = 'content'
    else:
        chunk_size = 800 # validation (800 x 10)
        text = 'comment_text'
    
    d={}
    for idx, row, lg in tqdm( zip(df_tmp["id"].values,
                                  df_tmp[text].values, 
                                  df_tmp["lang"].values) ):
            
            d[idx] = translate_blob(row, lg) if lg !='en' else row

            if (idx+1) % chunk_size == 0: # save intermediate results with size of +"chunk_size"
                df_tmp['content_en'] = df_tmp.id.map(d)
                df_tmp.to_csv(f"{path_to_save}/{name}_en.csv", index=False)
                print(f"Chunk {idx // chunk_size} saved successfuly")
                
                if idx % (len(df)-1) == 0:
                    print("execution completed successfully!")

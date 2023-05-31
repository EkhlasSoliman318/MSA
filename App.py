import pandas as pd
import numpy as np
import sqlite3
import re 
import string
from itertools import product
from camel_tools.dialectid import DialectIdentifier

def clean_text(text):
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    text = "".join([char for char in text if char not in string.ascii_letters]).strip() #remove abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    text = re.sub(r'[0-9\(\)/]+', '', text)
    # text = text.translate(str.maketrans('', '', string.punctuation)) #remove !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

        
    text = text.strip()
    
    return text

def main ():

    data_path = '/home/ekhlas/ASR_competition/Round_1/Data/wiki_sqlite3.db'
    # Create your connection.
    cnx = sqlite3.connect(data_path)

    df = pd.read_sql_query("SELECT * FROM  articles", cnx)
    # remove numaric rows
    df = df.copy()
    new_df = df[~df.Content.str.contains(r'\d')]


    #clean text 
    cleaned_text = []
    for text in new_df.Content:
        text = clean_text(text)
        cleaned_text.append(text)


    
    did = DialectIdentifier.pretrained()


    predictions = did.predict(cleaned_text)
    MSA_sent = []
    MSA_pred = []
    for p ,sent in zip(predictions , cleaned_text):
        if p.top =='MSA':
            MSA_sent.append(sent)
            MSA_pred.append(p.top)
    print('MSA sentences : ' , MSA_sent)
    # ##save csv file with output
    # df_mSA = pd.DataFrame(product(MSA_pred, MSA_sent), columns= (MSA_pred,MSA_sent))
    # df_mSA.to_csv(data_path + 'MSA_sentences.csv')


if __name__ == "__main__":
    main()
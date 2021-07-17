import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from googletrans import Translator
from nltk.tokenize import TreebankWordTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

#To convert Devanagiri words to Roman

def translate(words):
    words = transliterate(words,sanscript.DEVANAGARI, sanscript.ITRANS)
    return words

if __name__ == "__main__":
    file = input("Input manager excel file path: ").strip()
    df0 = pd.read_excel(file)

    #Pre-Processing

    df0['Sent'] = df0['Sent'].str.replace(r'[http,https]+\/\/\S*', '')
    tokenizer = TreebankWordTokenizer()
    df0['pro_sents'] = df0.apply(lambda row: tokenizer.tokenize(row['Sent']), axis=1)
    df0['pro_sents'] = df0['pro_sents'].apply(lambda x: list(map(translate, x)))
    df0['pro_sents'] = [' '.join(map(str, l)) for l in df0['pro_sents']]
    df0['pro_sents'] = df0['pro_sents'].str.replace(r"[^a-zA-Z\d\-.!,`'?]+", " ").str.lower()

    #Translation of entire sentence to English
    translator = Translator(service_urls=['translate.google.co.in'])
    df0['pro_sents'] = df0['pro_sents'].apply(translator.translate, src='hi', dest='en').apply(getattr, args=('text',))

    #Vader Sentiment Analysis

    analyzer = SentimentIntensityAnalyzer()
    sentiment = df0['pro_sents'].apply(lambda x: analyzer.polarity_scores(x))
    df0 = pd.concat([df0, sentiment.apply(pd.Series)], 1)
    conditions = [
        (df0['compound'] < -0.05) & (df0['compound'] >= -1),
        (df0['compound'] >= -0.05) & (df0['compound'] <= 0.05),
        (df0['compound'] > 0.05) & (df0['compound'] <= 1)
    ]

    values = ['negative', 'neutral', 'positive']

    df0['id'] = np.select(conditions, values)
    df0.to_excel(r'D:\Accelerate\Hinglish\ts.xlsx', header=False, index=False, columns=['Sent','id'])
    print("end")


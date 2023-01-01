import os

import Module1
import Module2
import Module3
from Module3 import *
from sklearn.feature_extraction.text import CountVectorizer

topdir = 'assets/iula'
all_content = []
for dirpath, dirnames, filename in os.walk(topdir):
    for name in filename:
        if name.endswith('plain.txt'):
            with open(os.path.join(dirpath, name)) as f:
                all_content.append(f.read())

processed_content = [Module3.preProcess(s) for s in all_content]

vectorizer = CountVectorizer()
vectorizer.fit(processed_content)
freq_iula = vectorizer.transform(processed_content)
freq_iula = pd.DataFrame(freq_iula.todense(), columns=vectorizer.get_feature_names()).sum()

COCA = pd.DataFrame([["this", 21940], ["code", 6], ['is', 3972], ['not', 1240], ['that', 2237], ['great', 0]],
                    columns=['word', 'frequency'])
COCA_pop = 1001610938
COCA['P(w)'] = COCA['frequency'] / COCA_pop
COCA['rank'] = COCA['frequency'].rank(ascending=False, method='min').astype(int)

WIKI = pd.DataFrame([['this', 121408], ['code', 81], ['is', 7793], ['not', 814], ['that', 1416], ['great', 0]],
                    columns=['word', 'frequency'])
WIKI_pop = 1.9e9
WIKI['P(w)'] = WIKI['frequency'] / WIKI_pop
WIKI['rank'] = WIKI['frequency'].rank(ascending=False, method='min').astype(int)

query = ['this', 'code', 'is', 'not', 'that', 'great']
transformed_query = [vectorizer.inverse_transform(vectorizer.transform([q])) for q in query]
query_freq = pd.Series([freq_iula.T.loc[tq[0]].values[0] if len(tq[0]) > 0 else 0 for tq in transformed_query],
                       index=query)
IULA = pd.DataFrame([["this", 11], ['code', 0], ['is', 198], ['not', 0], ['that', 15], ['great', 0]],
                    columns=['word', 'frequency'])
IULA_pop = 2.1e6
IULA['P(w)'] = IULA['frequency'] / IULA_pop
IULA['rank'] = IULA['frequency'].rank(ascending=False).astype(int)

norvig = pd.read_csv('http://norvig.com/ngrams/count_1edit.txt', sep='\t', encoding="ISO-8859-1", header=None)
norvig.columns = ['term', 'edit']
norvig = norvig.set_index('term')
print(norvig.head())

norvig_orig = pd.read_csv('http://norvig.com/ngrams/count_big.txt', sep='\t', encoding="ISO-8859-1", header=None)
norvig_orig = norvig_orig.dropna()
norvig_orig.columns = ['term', 'freq']
print(norvig_orig.head())


def get_count(c, norvig_orig):
    return norvig_orig.apply(lambda x: x.term.count(c) * x.freq, axis=1).sum()


from string import ascii_lowercase
from multiprocessing.pool import ThreadPool as Pool
import itertools

character_set = list(map(''.join, itertools.product(ascii_lowercase, repeat=1))) \
                + list(map(''.join, itertools.product(ascii_lowercase, repeat=2)))
pool = Pool(8)
freq_list = pool.starmap(get_count, zip(character_set, itertools.repeat(norvig_orig)))

freq_df = pd.DataFrame([character_set, freq_list], index=['char', 'freq']).T
freq_df = freq_df.set_index('char')

COCA['P(x|w)'] = [(norvig.loc['e|ea'].values / freq_df.loc['ea'].values)[0],
                  (norvig.loc['f|c'].values / freq_df.loc['c'].values)[0],
                  (norvig.loc['e|ec'].values / freq_df.loc['ec'].values)[0],
                  (norvig.loc['e| '].values / freq_df.loc['e'].values)[0],
                  (norvig.loc['t|r'].values / freq_df.loc['r'].values)[0],
                  (norvig.loc['fe|ef'].values / freq_df.loc['ef'].values)[0]]

COCA['109 P(x|w)P(w)'] = 1e9 * COCA['P(w)'] * COCA['P(x|w)']
COCA['109 P(x|w)P(w)']

IULA['P(x|w)'] = COCA['P(x|w)']
IULA['109 P(x|w)P(w)'] = 1e9 * IULA['P(w)'] * IULA['P(x|w)']


# Module3
from sklearn.feature_extraction.text import TfidfVectorizer

import Module1
import Module2
from Module2 import*

def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    s = [w for w in s if w not in stopwords_set]
    #s = [w for w in s if not w.isdigit()]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s

def sk_vectorize():
    cleaned_description = Module1.get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess)
    vectorizer.fit(cleaned_description)
    query = vectorizer.transform(['good at java and python'])
    print(query)
    print(vectorizer.inverse_transform(query))

# sk_vectorize()

vectorizer = CountVectorizer(preprocessor=preProcess,ngram_range=(1, 2))
X = vectorizer.fit_transform(cleaned_description)
print(vectorizer.get_feature_names())




N = 5
cleaned_description = Module1.get_and_clean_data()
cleaned_description = cleaned_description.iloc[:N]
vectorizer = CountVectorizer(preprocessor=preProcess)
X = vectorizer.fit_transform(cleaned_description)
print(X.toarray())

df = np.array((X.todense()>0).sum(0))[0]
idf = np.log10(N / df)
tf = np.log10(X.todense()+1)
tf_idf = np.multiply(tf, idf)
X = sparse.csr_matrix(tf_idf)

print(X.toarray())
print(pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names()))

arr = np.array([[100, 90, 5], [200, 200, 200], [200, 300, 10], [50, 0, 200]])

data = pd.DataFrame(arr, columns=['DH', 'CD', 'DC'],
index=['business', 'computer', 'git', 'parallel'])
data = np.log10(data + 1)

print(data['DH'].dot(data['CD']))
print(data['DH'].dot(data['DC']))
print(data['CD'].dot(data['DC']))

data['DH'] /= np.sqrt((data['DH'] ** 2).sum())
data['CD'] /= np.sqrt((data['CD'] ** 2).sum())
data['DC'] /= np.sqrt((data['DC'] ** 2).sum())

print(data.to_markdown())

print(data['DH'].dot(data['CD']))
print(data['DH'].dot(data['DC']))
print(data['CD'].dot(data['DC']))

N = 5
cleaned_description = Module1.get_and_clean_data()
cleaned_description = cleaned_description.iloc[:N]
vectorizer = TfidfVectorizer(preprocessor=preProcess)
X = vectorizer.fit_transform(cleaned_description)
print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))



#N = 5
#cleaned_description = Module1.get_and_clean_data()
#leaned_description = cleaned_description.iloc[:N]
#bm25 = BM25()
#bm25.fit(cleaned_description)
#print(bm25.transform('aws github',cleaned_description))





class BM25(object):
    def __init__(self, vectorizer, b=0.75, k1=1.6):
        self.vectorizer = vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        self.y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        len_y = self.y.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        y = self.y.tocsc()[:, q.indices]
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = y.multiply(np.broadcast_to(idf, y.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

cleaned_description = Module1.get_and_clean_data()
tf_idf_vectorizer = TfidfVectorizer(preprocessor=preProcess)
bm25 = BM25(tf_idf_vectorizer)
bm25.fit(cleaned_description)

score = bm25.transform('aws devops')
rank = np.argsort(score)[::-1]
print(cleaned_description.iloc[rank[:5]].to_markdown())

score = bm25.transform("aws github")
rank = np.argsort(score)[::-1]
print(cleaned_description.iloc[rank[:5]].to_markdown())
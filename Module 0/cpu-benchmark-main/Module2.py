from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy.ma.bench import m1
from scipy import sparse
from scipy.sparse import dok_matrix, lil_matrix, coo_matrix, csc_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import timeit
from Module1 import *
import Module1

cleaned_description = get_and_clean_data()
cleaned_description = cleaned_description.iloc[:4]

tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
sw_removed_description = tokenized_description.apply(lambda s: [word for word in s if word not in stopwords.words()])
sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])

ps = PorterStemmer()
stemmed_description = sw_removed_description.apply(lambda s: [ps.stem(w) for w in s])

cv = CountVectorizer(analyzer=lambda x: x)
X = cv.fit_transform(stemmed_description)

print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names()))

cleaned_description = get_and_clean_data()
cleaned_description = cleaned_description.iloc[:20]

tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
stop_dict = {s: 1 for s in stopwords.words()}
sw_removed_description = tokenized_description.apply(lambda s: [word for word in s if word not in stop_dict])
sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])

ps = PorterStemmer()
stemmed_description = sw_removed_description.apply(lambda s: [ps.stem(w) for w in s])

cv = CountVectorizer(analyzer=lambda x: x)
X = cv.fit_transform(stemmed_description)

# print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names()))
# print(X.tocsr()[0,:])

#timeit.timeit(lambda: np.matmul(X.toarray(), X.toarray().T), number=1)
#np.shape(np.matmul(X.toarray(), X.toarray().T))
#timeit.timeit(lambda: X * X.T, number=1)
#np.shape(X * X.T)

print(X.tocsr()[0, :])
csr = (timeit.timeit(lambda: X * X.T, number=1))
dok = (timeit.timeit(lambda: X.todok() * X.T.todok(), number=1))
lil = (timeit.timeit(lambda: X.tolil() * X.T.tolil(), number=1))
coo = (timeit.timeit(lambda: X.tocoo() * X.T.tocoo(), number=1))
csc = (timeit.timeit(lambda: X.tocsc() * X.T.tocsc(), number=1))

B = X.todense()
times = 100

compCsr = (timeit.timeit(lambda: csr_matrix(B), number=times) / times)
compDok = (timeit.timeit(lambda: dok_matrix(B), number=times) / times)
compLil = (timeit.timeit(lambda: lil_matrix(B), number=times) / times)
compCoo = (timeit.timeit(lambda: coo_matrix(B), number=times) / times)
compCsc = (timeit.timeit(lambda: csc_matrix(B), number=times) / times)

totalCsr = csr + compCsr
totalDok = dok + compDok
totalLil = lil + compLil
totalCoo = coo + compCoo
totalCsc = csc + compCsc

print('csr : ', csr)
print('dok : ', dok)
print('lil : ', lil)
print('coo : ', coo)
print('csc : ', csc)

print('compression csr : ', compCsr)
print('compression dok : ', compDok)
print('compression lil : ', compLil)
print('compression coo : ', compCoo)
print('compression csc : ', compCsc)

print('total csr : ', totalCsr)
print('total dok : ', totalDok)
print('total lil : ', totalLil)
print('total coo : ', totalCoo)
print('total csc : ', totalCsc)

#B = X.todense()
#times = 100
#
#print('compression csr : ', timeit.timeit(lambda: csr_matrix(B), number=times) / times)
#print('compression dok : ', timeit.timeit(lambda: dok_matrix(B), number=times) / times)
#print('compression lil : ', timeit.timeit(lambda: lil_matrix(B), number=times) / times)
#print('compression coo : ', timeit.timeit(lambda: coo_matrix(B), number=times) / times)
#print('compression csc : ', timeit.timeit(lambda: csc_matrix(B), number=times) / times)


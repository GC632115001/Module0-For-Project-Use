from flask import Flask, request
from scipy.sparse import hstack
from Module7 import *
import pickle

app = Flask(__name__)
app.vecterizer = pickle.load(open('resource/github_bug_prediction_tfidf_vectorizer.pkl', 'rb'))
app.model = pickle.load(open('resource/github_bug_prediction_model.pkl', 'rb'))
app.stopword_set = set(stopwords.words())
app.stemmer = PorterStemmer()


@app.route('/predict', methods=['GET'])
def search():
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    title = argList['title'][0]
    body = argList['body'][0]
    predict = app.model.predict_proba(
        hstack([app.vecterizer.transform([preprocess(title, app.stopword_set, app.stemmer)])]))
    response_object['predict_as'] = 'bug' if 1 - predict[0][1] >= 0.5 else 'not bug'
    response_object['bug_prob'] = 1 - predict[0][1]
    return response_object


if __name__ == '__main__':
    app.run(debug=True)

# Latent Sematic Analysis

from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(n_components=500, n_iter=100, random_state=0)
lsa.fit(X_tfidf_fit)
X_lsa_fit = lsa.transform(X_tfidf_fit)

gbm_model_with_lsa = lgb.LGBMClassifier()

precision_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5, n_jobs=-2,
                                                     scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()

print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))

X_fit_with_lsa = hstack([X_tfidf_fit, X_lsa_fit]).tocsr()

precision_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5, n_jobs=-2,
                                                     scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()

print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))

from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(n_components=500, n_iter=100, random_state=0)
lsa.fit(X_tfidf_fit)
X_lsa_fit = lsa.transform(X_tfidf_fit)

gbm_model_with_lsa = lgb.LGBMClassifier()

precision_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5, n_jobs=-2,
                                                     scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()

print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))

X_fit_with_lsa = hstack([X_tfidf_fit, X_lsa_fit]).tocsr()

precision_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5, n_jobs=-2,
                                                     scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()

print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))



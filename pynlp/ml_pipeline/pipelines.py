from sklearn.pipeline import Pipeline, FeatureUnion
from ml_pipeline import preprocessing, representation
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


def pipeline(preprocessor, representation, classifier):
    return Pipeline([('prep', preprocessor),
                     ('frm', representation),
                     ('clf', classifier)])


def combined_pipeline(prep1, repr1, prep2, repr2, classifier):
    combined_features = FeatureUnion([
        ('token_features', Pipeline([('prep1', prep1), ('repr1', repr1)])),
        ('polarity_features', Pipeline([('prep2', prep2), ('repr2', repr2)]))])
    return Pipeline([('features', combined_features),
                     ('clf', classifier)])


# ------------- parametrization ---------------------------


def svm_clf_grid_parameters():
    """Example parameters for svm.LinearSVC grid search

    The preprocessor and formatter can also be parametrized through the prefixes 'prep' and 'frm', respectively."""
    return {'clf__class_weight': (None, 'balanced'),
            'clf__dual': (True, False),
            'clf__C': (0.1, 1, 10)}


# ------------- standard pipelines ---------------------------------
def naive_bayes_counts():
    return pipeline(preprocessing.std_prep(), representation.count_vectorizer({'min_df': 1}), MultinomialNB())


def naive_bayes_tfidf():
    return pipeline(preprocessing.std_prep(), representation.tfidf_vectorizer(), MultinomialNB())


def svm_libsvc_counts():
    return pipeline(preprocessing.std_prep(), representation.count_vectorizer(), svm.LinearSVC(max_iter=10000,
                                                                                               dual=False, C=0.1))


def svm_libsvc_tfidf():
    return pipeline(preprocessing.std_prep(), representation.tfidf_vectorizer(), svm.LinearSVC(max_iter=10000,
                                                                                               dual=False, C=0.1))


def svm_libsvc_embed():
    return pipeline(preprocessing.std_prep(), representation.text2embeddings('wiki-news'), svm.LinearSVC(max_iter=10000,
                                                                                              dual=False, C=0.1))


def svm_sigmoid_embed():
    return pipeline(preprocessing.std_prep(), representation.text2embeddings('glove'), svm.SVC(kernel='sigmoid',
                                                                                        gamma='scale'))


def naive_bayes_counts_lex():
    # This method is copy pasted from Canvas. See:
    # https://canvas.vu.nl/courses/63973/pages/pipeline-7-use-a-lexicon-for-preprocessing
    #
    # TODO: Why is there also a naive_bayes_counts_lex method in pipeline_with_lexicon ??
    return pipeline(preprocessing.lex_prep(), representation.count_vectorizer({'min_df': 1}), MultinomialNB())


# ------------- added pipelines ---------------------------------

def svm_libsvc_counts_lex():
    return pipeline(preprocessing.lex_prep(), representation.count_vectorizer(), svm.LinearSVC(max_iter=10000,
                                                                                               dual=False, C=0.1))


def svm_libsvc_tfidf_lex():
    return pipeline(preprocessing.lex_prep(), representation.tfidf_vectorizer(), svm.LinearSVC(max_iter=10000,
                                                                                               dual=False, C=0.1))


def svm_sigmoid_embed_lex():
    return pipeline(preprocessing.lex_prep(), representation.text2embeddings('glove'), svm.SVC(kernel='sigmoid',
                                                                                        gamma='scale'))

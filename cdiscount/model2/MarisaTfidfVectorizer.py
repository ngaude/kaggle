import marisa_trie
from sklearn.externals import six
from sklearn.feature_extraction.text import TfidfVectorizer

class MarisaTfidfVectorizer(TfidfVectorizer):
    def fit_transform(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit_transform(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit_transform(raw_documents, y)
    def fit(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit(raw_documents, y)
    def _freeze_vocabulary(self, X=None):
        if not self.fixed_vocabulary_:
            self.vocabulary_ = marisa_trie.Trie(six.iterkeys(self.vocabulary_))
            self.fixed_vocabulary_ = True
            del self.stop_words_



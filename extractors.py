import numpy as np
import abc
from collections import Counter, OrderedDict
import hashlib
import re

class Extractor(metaclass=abc.ABCMeta):
    # Base class Extractor subclass.
    def __init__(self, argument):
        self.argument = argument

    @abc.abstractmethod
    def extract_features(self, sentences):
        raise NotImplementedError("Must be subclassed.")

class EmptyData():
    def __init__(self):
        self.vocab = {}
        self.vector_size = 300
        
class RandomVectors(Extractor):
    def __init__(self, dimensionality, argument, **kwargs):
        super().__init__(argument)
        self.vocab = {}
        self.n_features = dimensionality

    def extract_features(self, sentence):
        feats = np.array([self[w] for w in sentence])
        assert feats.shape == (len(sentence), self.n_features)
        return feats

    def __getitem__(self, w):
        if w not in self.vocab:
            # Setting hashing state ensures we have the same random vector for each word between runs
            hsh = hashlib.md5()
            hsh.update(w.encode())
            seed = int(hsh.hexdigest(), 16) % 4294967295  # highest number allowed by seed
            state = np.random.RandomState(seed)  # pylint: disable=E1101
            self.vocab[w] = state.randn(self.n_features)  # pylint: disable=E1101
        return self.vocab[w]
    
_word2vec_data = None

class Word2Vec(Extractor):
    # Loads Word2Vec vectors from binary file format, as created by the Google News corpus
    def __init__(self, path, argument, **kwargs):
        super().__init__(argument)
        self.path = path
        self.data = self._load_from_binary_format(self.path)

        self.n_embeddings = len(self.data.vocab)
        self.n_features = self.data.vector_size
        self.random_vectors = RandomVectors(self.n_features, argument)

    def extract_features(self, sentence):
        # Returns features according to: n_words x n_features
        feats = np.array([self[w] for w in sentence])
        assert feats.shape == (len(sentence), self.n_features)
        return feats


    def _load_from_binary_format(self, path):
        global _word2vec_data  # pylint: disable=W0603
        if _word2vec_data is None:
            import gensim
            _word2vec_data = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            
        return _word2vec_data

    def __getitem__(self, w):
        if w in self.data.vocab:
            return self.data[w]
        else:
            return self.random_vectors[w]


class CBOW(Word2Vec):
    def __init__(self, path, argument, **kwargs):
        super().__init__(path, argument, **kwargs)
        self.name = CBOW

    def extract_features(self, sentence):
        feats = np.mean(super().extract_features(sentence), axis=0, keepdims=True)
        assert feats.shape == (1, self.n_features)
        return feats
        
class OneHot(Extractor):
    def __init__(self, instances, max_vocab_size, argument):
        super().__init__(argument)
        self.vocab = self._read_vocab_indices_from_input(instances, max_vocab_size, argument)
        self.n_features = len(self.vocab) + 1 # last one for oov

    def _read_vocab_indices_from_input(self, instances, max_vocab_size, argument):
        counts = Counter()
        words = re.compile(r'\w+')
        vocab = {}
        for tag in argument:
            for i, rel in enumerate(instances):
                arg = getattr(rel, tag)()
                if tag is "arg1_text" or tag is "arg2_text":
                    counts.update(words.findall(arg.lower()))
                elif tag is "connective_token":
                    if (arg is not None):
                        counts.update({arg.lower():1})
                    else:
                        counts.update({arg:1})
        
        counts = OrderedDict(counts.most_common())
        i = 0
        for arg, count in counts.items():
            if count >= 1 and arg not in vocab:
                vocab[arg] = i
                i = i+1
        return vocab

    def get(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return len(self.vocab) # oov

    def extract_features(self, sentence):
        feat_matrix = np.zeros([len(sentence), self.n_features])
        for i, w in enumerate(sentence):
            feat_matrix[i, self.get(w)] = 1
        assert feat_matrix.shape == (len(sentence), self.n_features)
        return feat_matrix
        
class BagOfWords(OneHot):
    def __init__(self, data, max_vocab_size, argument, **kwargs):
        super().__init__(data, max_vocab_size, argument, **kwargs)
        self.name = BagOfWords

    def extract_features(self, sentence):
        feats = np.sum(super().extract_features(sentence), axis=0, keepdims=True)
        assert feats.shape == (1, self.n_features)
        return feats

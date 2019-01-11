from relations_utils import DiscourseRelation
from collections import Counter, defaultdict
import json
import abc
import numpy as np
import re

def tokenize(sentence):
    return sentence.lower()

class Resource(metaclass=abc.ABCMeta):
    def __init__(self, path, max_words_in_sentence, classes, padding):
        self.path = path
        self.max_words_in_sentence = max_words_in_sentence
        self.classes = sorted(classes)
        self.y_indices = {x: y for y, x in enumerate(self.classes)}
        self.instances = list(self._load_instances(path))
        self.padding = padding

    @abc.abstractmethod
    def _load_instances(self, path):
        raise NotImplementedError("This class must be subclassed.")


class PDTBRelations(Resource):
    def __init__(self, path, max_words_in_sentence, max_hierarchical_level, classes, separate_dual_classes, padding, filter_type=[]):
        self.max_hierarchical_level = max_hierarchical_level
        self.separate_dual_classes = separate_dual_classes
        self.filter_type = filter_type
        super(PDTBRelations, self).__init__(path, max_words_in_sentence, classes, padding)

    def _load_instances(self, path):
        with open(path) as file_:
            for line in file_:
                rel = DiscourseRelation(json.loads(line.strip()))
                if self.separate_dual_classes:
                    for splitted in rel.split_up_senses(max_level=self.max_hierarchical_level):
                        yield splitted
                else:
                    yield rel

    def massage_sentence(self, sentence, extractor, tag):
        words = re.compile(r'\w+')
        if sentence is None:
            tokenized = ["NONE"]   
        elif tag is "arg1_text" or tag is "arg2_text":
            tokenized = words.findall(tokenize(sentence))
        else:
            tokenized = [tokenize(sentence)]

        if hasattr(extractor, 'sentence_max_length') and extractor.sentence_max_length:
            tokenized = tokenized[:extractor.sentence_max_length]
        if self.padding:
            tokenized = tokenized + ['PADDING'] * (extractor.sentence_max_length - len(tokenized))

        return tokenized

    def get_feature_tensor(self, extractors):
        rels_feats = []
        n_instances = 0
        last_features_for_instance = None
        for rel in self.instances:
            n_instances += 1
            feats = []
            total_features_per_instance = 0
            for extractor in extractors:
                for tag in extractor.argument:
                    # These return matrices of shape (1, n_features)
                    # We concatenate them on axis 1
                    arg_rawtext = getattr(rel, tag)()
                    arg_tokenized = self.massage_sentence(arg_rawtext, extractor, tag)
                    arg_feats = extractor.extract_features(arg_tokenized)
                    feats.append(arg_feats)
                    total_features_per_instance += extractor.n_features
            if last_features_for_instance is not None:
                # Making sure we have equal number of features per instance
                assert total_features_per_instance == last_features_for_instance
            rels_feats.append(np.concatenate(feats, axis=1))

        feature_tensor = np.array(rels_feats)
        assert_shape = (n_instances, 1, total_features_per_instance)
        assert feature_tensor.shape == assert_shape, \
                "Tensor shape mismatch. Is {}, should be {}".format(feature_tensor.shape, assert_shape)
        return feature_tensor

    def get_correct(self, indices=True):
        """
        Returns answer indices.
        """

        for rel in self.instances:
            senses = rel.senses(max_level=self.max_hierarchical_level)
            if self.separate_dual_classes:
                if indices:
                    yield self.y_indices[senses[0]]
                else:
                    yield senses[0]
            else:
                ys = [self.y_indices[sense] for sense in senses]
                if indices:
                    yield ys
                else:
                    yield senses

    def calculate_accuracy(self, predicted):
        equal = 0
        gold = list(self.get_correct(indices=True))
        assert len(predicted) == len(gold)
        for p, g in zip(predicted, gold):
            assert isinstance(g, list)
            if p in g:
                equal += 1
        return equal / len(predicted)

    def store_results(self, results, store_path):
        """
        Don't forget to use the official scoring script here.
        """
        text_results = [self.classes[res] for res in results]
        # Load test file
        # Output json object with results
        # Deal with multiple instances somehow
        predicted_rels = []
        for text_result, rel in zip(text_results, self.instances):
            if rel.is_explicit():
                rel_type = 'Explicit'
            else:
                rel_type = 'Implicit'
            predicted_rels.append(rel.to_output_format(text_result, rel_type))  # turn string representation into list instance first

        # Store test file
        with open(store_path, 'w') as w:
            for rel in predicted_rels:
                w.write(json.dumps(rel) + '\n')
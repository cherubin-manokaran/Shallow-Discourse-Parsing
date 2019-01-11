from Data_Reader import PDTBRelations
from extractors import BagOfWords, CBOW
from CNN import CNN
import numpy as np
import codecs

def get_answers(instances):
    return list(instances.get_correct())

def initialize_extractors(methods, path, instances, vocab_size, tag):
    extractors = []
    for method in methods:
        if method is BagOfWords:
            extractors.append(method(instances, vocab_size, tag))
        elif method is CBOW:
            extractors.append(method(path, tag))
    return extractors

def extract_features(extractors, instances):
    return instances.get_feature_tensor(extractors)

def parse():
    classes = ["Temporal", "Temporal.Asynchronous", "Temporal.Asynchronous.Precedence", "Temporal.Asynchronous.Succession", "Temporal.Synchrony",
              "Contingency", "Contingency.Cause", "Contingency.Cause.Reason","Contingency.Cause.Result", "Contingency.Condition", "Comparison",
              "Comparison.Contrast", "Comparison.Concession", "Expansion", "Expansion.Conjunction", "Expansion.Instantiation", 
              "Expansion.Restatement", "Expansion.Alternative", "Expansion.Alternative.Chosen alternative", "Expansion.Exception", 
              "EntRel"]
    
    # File Paths
    word2vec_path = "GoogleNews-vectors-negative300.bin"
    train_path = 'train/relations.json'
    dev_path = 'dev/relations.json'
    test_path = 'test/relations.json'
    out_path = 'output/output.json'
    store_path = 'models/cnn.ckpt'
    
    extraction_methods = [BagOfWords]
    tag = ["connective_token"]

    # Hyperparameters          
    l2_reg_lambda = 3
    filter_sizes = [1,2]
    num_filters = 128
    dropout_keep_prob = 0.5
    
    # Training Parameters
    max_hierarchical_level = 3
    max_words_in_sentence = 50
    batch_size = 64
    num_epochs = 10
    evaluate_every = 100
    checkpoint_every = 100
    vocab_size=300000
    
    separate_dual_classes = False
    padding = False
    
    training_data = PDTBRelations(path=train_path, max_words_in_sentence=max_words_in_sentence, 
                                  max_hierarchical_level=max_hierarchical_level, classes=classes, 
                                  separate_dual_classes=separate_dual_classes, padding=padding)
    dev_data = PDTBRelations(path=dev_path, max_words_in_sentence=max_words_in_sentence, 
                             max_hierarchical_level=max_hierarchical_level, classes=classes, 
                             separate_dual_classes=separate_dual_classes, padding=padding)
    
    # Feature Extraction
    extractors = initialize_extractors(extraction_methods, path=word2vec_path,
                                       instances=training_data.instances+dev_data.instances, 
                                       vocab_size=vocab_size, tag=tag)
    for extractor in extractors:
        if extractor.name is BagOfWords:
            vocab_size = len(extractor.vocab)
        
    train_correct = get_answers(training_data)
    train_extracted_features = extract_features(extractors, training_data)
    dev_correct = get_answers(dev_data)
    dev_extracted_features = extract_features(extractors, dev_data)
    
    # Initialize and Train Model
    model = CNN(n_features=train_extracted_features.shape[2], n_classes=len(training_data.y_indices),
                batch_size=batch_size, num_epochs=num_epochs, evaluate_every=evaluate_every, 
                checkpoint_every=checkpoint_every, max_words_in_sentence=train_extracted_features.shape[2],
                vocab_size=vocab_size, l2_reg_lambda=l2_reg_lambda, dropout_keep_prob=dropout_keep_prob, 
                filter_sizes=filter_sizes, num_filters=num_filters, store_path=store_path)
    model.train(training_data, dev_data, train_extracted_features, train_correct, 
                dev_extracted_features, dev_correct)
    
    test_data = PDTBRelations(test_path, max_words_in_sentence=max_words_in_sentence, 
                              max_hierarchical_level=max_hierarchical_level, classes=classes, 
                              separate_dual_classes=separate_dual_classes, padding=padding)

    # Test Model    
    test_extracted_features = extract_features(extractors, test_data)
    model = CNN(n_features=test_extracted_features.shape[2], n_classes=len(test_data.y_indices),
                batch_size=batch_size, num_epochs=num_epochs, evaluate_every=evaluate_every, 
                checkpoint_every=checkpoint_every, max_words_in_sentence=test_extracted_features.shape[2], 
                vocab_size=vocab_size, l2_reg_lambda=l2_reg_lambda, dropout_keep_prob=dropout_keep_prob, 
                filter_sizes=filter_sizes, num_filters=num_filters, store_path=store_path)
    predicted = model.test(test_extracted_features)
    
    print("Accuracy: {:g}".format(test_data.calculate_accuracy(predicted)))
    test_data.store_results(predicted, out_path)
    
if __name__ == '__main__':
    parse()
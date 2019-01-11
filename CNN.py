import tensorflow as tf
import numpy as np
import os
import time
import datetime

def test_cnn():
    return CNN()

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))  # pylint:disable=E1101
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

class CNN():
    def __init__(self, n_features=20, n_classes=3, embedding_dim=300, filter_sizes=[3,4,5],
                 num_filters=128, dropout_keep_prob=0, l2_reg_lambda=0, batch_size=64,
                 num_epochs=10, evaluate_every=100, checkpoint_every=100, allow_soft_placement=True,
                 log_device_placement=False, max_words_in_sentence=20, vocab_size=300000,
                 store_path=None, word2vec_path=None, name=None):

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.max_words_in_sentence = max_words_in_sentence
        self.n_features = n_features
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.graph = tf.Graph()
        self.store_path = store_path
        self.word2vec_path = word2vec_path
        if self.word2vec_path is not None:
            self.embedding_matrix = Word2Vec(self.word2vec_path, 'connective_token').data.syn0[:self.vocab_size]
        else:
            self.embedding_matrix = np.random.randn(vocab_size, embedding_dim)

        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=self.allow_soft_placement,
                                          log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            self.sess = sess
            with self.sess.as_default():
                self.cnn = TextCNN(sequence_length=self.max_words_in_sentence, num_classes=self.n_classes,
                                   vocab_size=self.vocab_size, embedding_size=self.embedding_dim, 
                                   filter_sizes=self.filter_sizes, num_filters=self.num_filters, 
                                   l2_reg_lambda=self.l2_reg_lambda)

                # Define Training procedure
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdagradOptimizer(0.01)
                grads_and_vars = optimizer.compute_gradients(self.cnn.loss)
#                capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) \
#                              if grad != None else (grad, var) for grad, var in grads_and_vars]
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", self.cnn.accuracy)

                # Train Summaries
                self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                self.saver = tf.train.Saver(tf.global_variables())

                # Initialize all variables
#                feed = {self.cnn.embeddings_placeholder: self.embedding_matrix}
                sess.run(tf.global_variables_initializer())

    def train_step(self, x_batch, y_batch):
        feed_dict = {
          self.cnn.input_x: x_batch,
          self.cnn.input_y: y_batch,
          self.cnn.dropout_keep_prob: self.dropout_keep_prob,
          self.cnn.embeddings_placeholder: self.embedding_matrix
        }
        _, step, summaries, loss, accuracy = self.sess.run([self.train_op, self.global_step,
                                                            self.train_summary_op, self.cnn.loss, 
                                                            self.cnn.accuracy], feed_dict)
        print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def dev_step(self, x_batch, y_batch, writer=None):
        with self.graph.as_default(), self.sess.as_default():
            feed_dict = {
              self.cnn.input_x: x_batch,
              self.cnn.input_y: y_batch,
              self.cnn.dropout_keep_prob: 1.0,
              self.cnn.embeddings_placeholder: self.embedding_matrix
            }
            step, summaries, loss, accuracy = self.sess.run([self.global_step, 
                                                             self.dev_summary_op, 
                                                             self.cnn.loss, 
                                                             self.cnn.accuracy],
                                                            feed_dict)
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

    def train(self, train_data, dev_data, x_train, y_train, x_dev, y_dev):
        x_train = np.squeeze(x_train, axis=1)
        y_train = self.massage_answers(train_data, y_train)
        x_dev = np.squeeze(x_dev, axis=1)
        y_dev = self.massage_answers(dev_data, y_dev)
        with self.graph.as_default(), self.sess.as_default():
            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.train_step(x_batch, y_batch)

                current_step = tf.train.global_step(self.sess, self.global_step)
                if current_step % self.evaluate_every == 0:
                    print("\nEvaluation:")
                    self.dev_step(x_dev, y_dev, writer=self.dev_summary_writer)
                if current_step % self.checkpoint_every == 0:
                    pass
                
            if self.store_path:
                self.store(self.store_path, self.sess)

    def test(self, feature_tensor):
        feature_tensor = np.squeeze(feature_tensor, axis=1)
        with self.graph.as_default(), self.sess.as_default():
            x_test = feature_tensor
            self.restore(self.store_path, self.sess)

            feed_dict = {
              self.cnn.input_x: x_test,
              self.cnn.dropout_keep_prob: 1.0,
              self.cnn.embeddings_placeholder: self.embedding_matrix
            }
            step, predictions = self.sess.run([self.global_step, self.cnn.predictions], feed_dict)
        return predictions

    def massage_answers(self, data, correct):
        if data.separate_dual_classes:
            labels_dense = np.array(correct)
            num_labels = labels_dense.shape[0]
            index_offset = np.arange(num_labels) * self.n_classes
            labels_one_hot = np.zeros((num_labels, self.n_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        else:
            labels_dense = np.array(correct)
            num_labels = labels_dense.shape[0]
            labels_one_hot = np.zeros((num_labels, self.n_classes))
            for i, labels in enumerate(labels_dense):
                for j, label in enumerate(labels):
                    labels_one_hot[i,label] = 1
        return labels_one_hot

    def restore(self, store_path, session):
        saver = tf.train.Saver()
        saver.restore(session, store_path)

    def store(self, store_path, session):
        os.makedirs(os.path.join(*store_path.split("/")[:-1]), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(session, store_path)

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings_placeholder = tf.placeholder(tf.float32, shape=(vocab_size, embedding_size))
            self.embedded_chars = tf.nn.embedding_lookup(self.embeddings_placeholder, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape = [num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



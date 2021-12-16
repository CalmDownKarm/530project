import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import datamanager

import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflowf.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# helper methods
def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_length_tweet = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        # all tweets are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_tweet)
        sequence_padded += [sp]
        sequence_length += [sl]
    return sequence_padded, sequence_length


def minibatches(data, labels, batch_size):
    data_size = len(data)
    start_index = 0

    num_batches_per_epoch = int((len(data) + batch_size - 1) / batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index: end_index], labels[start_index: end_index]
        
def select(parameters, length):
    """Select the last valid time step output as the sentence embedding
    :params parameters: [batch, seq_len, hidden_dims]
    :params length: [batch]
    :Returns : [batch, hidden_dims]
    """
    shape = tf.shape(parameters)
    idx = tf.range(shape[0])
    idx = tf.stack([idx, length - 1], axis = 1)
    return tf.gather_nd(parameters, idx)


dm = datamanager.DataManager()
X,y = dm.tokenize()
X_train, X_TEST, y_train, Y_TEST = train_test_split(X, y, test_size=0.2, random_state=0)
X_TRAIN, X_DEV, Y_TRAIN, Y_DEV = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Global variables
hidden_size_lstm_1 = 200
hidden_size_lstm_2 = 300
tags = 39
tweet_dim = 300
proj1 = 200
proj2 = 100
tweets = 20001
batchSize = 2
log_dir = "train"
model_dir = "DAModel"
model_name = "ckpt"

# Dialogue Act Recognition Model
# Architecture: dataset --> embedding --> utterance-level bi-LSTM --> conversation-level bi-LSTM --> CRF --> one label per utterance
class NBAModel():
    def __init__(self):
        with tf.variable_scope("placeholder"):
            self.num_games = tf.placeholder(tf.int32, shape=[None], name="num_games")
            self.tweets = tf.placeholder(tf.int32, shape=[None, None, None], name="tweets")
            self.tweet_lengths = tf.placeholder(tf.int32, shape=[None, None], name="tweet_lengths")
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
            self.clip = tf.placeholder(tf.float32, shape=[], name='clip')
          
        with tf.variable_scope("embeddings"):
            _tweet_embeddings = tf.get_variable(
                name = "_tweet_embeddings",
                dtype = tf.float32,
                shape = [tweets, tweet_dim],
                initializer = tf.random_uniform_initializer()
            )
            tweet_embeddings = tf.nn.embedding_lookup(_tweet_embeddings, self.tweets, name="tweet_embeddings")
            self.tweet_embeddings = tf.nn.dropout(tweet_embeddings, 0.8)
                    
        with tf.variable_scope("encoder"):
            s = tf.shape(self.tweet_embeddings)
            batch_size = s[0] * s[1]
            time_step = s[-2]

            tweet_embeddings = tf.reshape(self.tweet_embeddings, [batch_size, time_step, tweet_dim])
            length = tf.reshape(self.tweet_lengths, [batch_size])

            fw = tf.nn.rnn_cell.LSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple=True)
            bw = tf.nn.rnn_cell.LSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple=True)
            
            output, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, tweet_embeddings,sequence_length=length, dtype=tf.float32)
            output = tf.concat(output, axis=-1) # [batch_size, time_step, dim]

            # Select the last valid time step output as the utterance embedding, 
            # this method is more concise than TensorArray with while_loop
            output = tf.nn.avg_pool2d(output, ksize=(50,50), padding='VALID')  #select(output, length) # [batch_size, dim]
            output = tf.reshape(output, (s[0], s[1], 2 * hidden_size_lstm_1))
            output = tf.nn.dropout(output, 0.8)
                
        with tf.variable_scope("proj1"):
            output = tf.reshape(outputs, [-1, 2 * hidden_size_lstm_2])
            W = tf.get_variable("W", dtype=tf.float32, shape=[2 * hidden_size_lstm_2, proj1], initializer=tf.keras.initializers.glorot_uniform())
            b = tf.get_variable("b", dtype=tf.float32, shape=[proj1], initializer=tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(output, W) + b)

        #with tf.variable_scope("proj2"):
        #    W = tf.get_variable("W", dtype=tf.float32, shape=[proj1, proj2], initializer=tf.keras.initializers.glorot_uniform())
        #    b = tf.get_variable("b", dtype=tf.float32, shape =[proj2], initializer=tf.zeros_initializer())
        #    output = tf.nn.relu(tf.matmul(output, W) + b)

        with tf.variable_scope("logits"):
            nstep = tf.shape(outputs)[1]
            W = tf.get_variable("W", dtype=tf.float32, shape=[proj2, tags], initializer=tf.random_uniform_initializer())
            b = tf.get_variable("b", dtype=tf.float32, shape =[tags], initializer=tf.zeros_initializer())
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nstep, tags])

        with tf.variable_scope("loss"):
            transition_params = tf.get_variable("transitions", dtype=tf.float32, shape=[tags, tags])
            sequence_scores = tfa.text.crf_sequence_score(self.logits, self.labels, self.num_games, transition_params)
            log_norm = tfa.text.crf_log_norm(self.logits, self.dialogue_lengths, transition_params)
            log_likelihood = sequence_scores - log_norm
            self.trans_params = transition_params
            self.loss = tf.reduce_mean(-log_likelihood) + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)        

        with tf.variable_scope("viterbi_decode"):
            viterbi_sequence, _ = tfa.text.crf_decode(self.logits, self.trans_params,  self.num_games)
            batch_size = tf.shape(self.num_games)[0]
            output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
            
            def body(time, output_ta_1):
                length = self.num_games[time]
                vcode = viterbi_sequence[time][:length]
                true_labs = self.labels[time][:length]
                accurate = tf.reduce_sum(tf.cast(tf.equal(vcode, true_labs), tf.float32))

                output_ta_1 = output_ta_1.write(time, accurate)
                return time + 1, output_ta_1


            def condition(time, output_ta_1):
                return time < batch_size


            i = 0
            [time, output_ta] = tf.while_loop(condition, body, loop_vars=[i, output_ta])
            output_ta = output_ta.stack()
            accuracy = tf.reduce_sum(output_ta)
            self.accuracy = accuracy / tf.reduce_sum(tf.cast(self.num_games, tf.float32))

        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdagradOptimizer(0.1)
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))


def main():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Session(config=config) as sess:
        model = NBAModel()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("train", sess.graph)
        clip = 2
        counter = 0

        for epoch in range(100):
            for train_batch_X, train_batch_y in minibatches(X_TRAIN, Y_TRAIN, batchSize):
                #_, train_batch_lengths = pad_sequences(train_batch_X, 0)
                train_batch_tweets, train_batch_tweet_lengths = pad_sequences(train_batch_X, 0)
                #true_labs = train_batch_y
                counter += 1
                train_loss, train_accuracy, _ = sess.run(
                    [model.loss, model.accuracy,model.train_op],
                    feed_dict = {
                      model.tweets: train_batch_tweets, 
                      model.tweet_lengths: train_batch_tweet_lengths,
                      #model.num_games: train_batch_num_games,
                      model.labels: train_batch_y,
                      model.clip: clip
                    } 
                )
                print("step = {}, train_loss = {}, train_accuracy = {}".format(counter, train_loss, train_accuracy))
                
                train_precision_summ = tf.Summary()
                train_precision_summ.value.add(tag='train_accuracy', simple_value=train_accuracy)
                writer.add_summary(train_precision_summ, counter)

                train_loss_summ = tf.Summary()
                train_loss_summ.value.add(tag='train_loss', simple_value=train_loss)
                writer.add_summary(train_loss_summ, counter)
                
                if counter % 10 == 0:
                    dev_loss = []
                    dev_acc = []

                    for dev_batch_X, dev_batch_y in minibatches(X_DEV, Y_DEV, batchSize):
                        #_, dev_batch_num_games = pad_sequences(dev_batch_dialogues, 0)
                        dev_batch_tweets, dev_batch_tweets_lengths = pad_sequences(dev_batch_X, 0)
                        dev_batch_loss, dev_batch_acc = sess.run(
                          [model.loss, model.accuracy], 
                          feed_dict = {
                            model.tweets: dev_batch_tweets,
                            model.tweet_lengths: dev_batch_tweet_lengths,
                            #model.num_games: dev_batch_num_games,
                            model.labels: dev_batch_y,
                            model.clip: clip
                          }
                        )
                        dev_loss.append(dev_batch_loss)
                        dev_acc.append(dev_batch_acc)

                    valid_loss = sum(dev_loss) / len(dev_loss)
                    valid_accuracy = sum(dev_acc) / len(dev_acc)

                    dev_precision_summ = tf.Summary()
                    dev_precision_summ.value.add(tag='dev_accuracy', simple_value=valid_accuracy)
                    writer.add_summary(dev_precision_summ, counter)

                    dev_loss_summ = tf.Summary()
                    dev_loss_summ.value.add(tag='dev_loss', simple_value=valid_loss)
                    writer.add_summary(dev_loss_summ, counter)
                    print("counter = {}, dev_loss = {}, dev_accuacy = {}".format(counter, valid_loss, valid_accuracy))

        test_losses = []
        test_accs = []
        for test_batch_X, test_batch_y in minibatches(X_TEST, Y_TEST, batchSize):
            #_, test_batch_num_games = pad_sequences(test_batch, 0)
            test_batch_tweets, test_batch_tweet_lengths = pad_sequences(test_batch_X, 0)
            test_batch_loss, test_batch_acc = sess.run(
                [model.loss, model.accuracy],
                feed_dict={
                    model.tweets: test_batch_tweets,
                    model.tweet_lengths: test_batch_tweet_lengths,
                    #model.num_games: test_batch_num_games,
                    model.labels: test_batch_y,
                    model.clip: clip
                }
            )
            test_losses.append(test_batch_loss)
            test_accs.append(test_batch_acc)


if __name__ == "__main__":
    main()

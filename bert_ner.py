import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import numpy as np
from modeling import BertModel, BertConfig
import tokenization
import optimization
import sys
import os

bert_root = "/scratch/sanjay/cased_L-12_H-768_A-12"
conll2003_root = "/shared/corpora/ner/conll2003/eng/"
# bert_config = modeling.BertConfig.from_json_file("/scratch/sanjay/cased_L-24_H-1024_A-16/bert_config.json")
# bert_checkpoint = "/scratch/sanjay/cased_L-24_H-1024_A-16/bert_model.ckpt"
bert_config = os.path.join(bert_root, "bert_config.json")
bert_checkpoint = os.path.join(bert_root, "bert_model.ckpt")
vocab_file = os.path.join(bert_root, "vocab.txt")
entity_types_list = ['B-LOC', 'B-MISC', 'B-ORG', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O', 'X']
entity_types = {t: i for i, t in enumerate(entity_types_list)}

class NerModel:
    def __init__(self):
        self.input_ids = tf.placeholder(tf.int32, [None, None])
        self.input_mask = tf.placeholder(tf.int32, [None, None])
        self.model = BertModel(config=BertConfig.from_json_file(bert_config),
                               is_training=True,
                               input_ids=self.input_ids,
                               input_mask=self.input_mask)
        self.is_training = tf.placeholder(tf.bool, [])
        self.predictions = self.construct_model(self.model)
        self.id_predictions = tf.argmax(self.predictions, axis=2)
        self.Y = tf.placeholder(tf.float32, [None, None, len(entity_types)])
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=False)
        self.model_path = "/scratch/sanjay/bert/bert_ner_model.ckpt"
        self.saver = tf.train.Saver()

    def restore_model(self, sess):
        self.saver.restore(sess, self.model_path)
        
    def construct_model(self, bert_model):
        last_hidden_layer = bert_model.get_sequence_output()
        last_hidden_layer = tf.nn.dropout(last_hidden_layer, keep_prob=1-tf.cond(self.is_training, lambda: 0.1, lambda: 0.0))
        hidden_size = last_hidden_layer.shape[-1].value
        # output_layer = tf.get_variable("output_weights", shape=[hidden_size, len(entity_types)])
        # output_bias = tf.get_variable("output_bias", shape=[len(entity_types)])
        output_weights = tf.Variable(tf.random_normal([hidden_size, len(entity_types)]))
        output_bias = tf.Variable(tf.random_normal([len(entity_types)]))
        predictions = tf.matmul(tf.reshape(last_hidden_layer, [-1, hidden_size]), output_weights)
        predictions = tf.nn.bias_add(predictions, output_bias)
        predictions = tf.nn.log_softmax(predictions)
        predictions = tf.reshape(predictions, [tf.shape(last_hidden_layer)[0], tf.shape(last_hidden_layer)[1], output_weights.shape[-1].value])
        return predictions

    def process_conll_data(self, fname, is_training=True):
        f = open(fname)
        lines = f.readlines()
        ids = []
        labels = []
        curr_sent_ids = []
        curr_sent_labels = []
        sentences = [[]]
        doc_start = False
        max_sent_length = 0
        max_desired_sent_length = 64
        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                if not doc_start:
                    if is_training:
                        while len(curr_sent_ids) > max_desired_sent_length:
                            index = max_desired_sent_length
                            while index < len(curr_sent_ids) and curr_sent_labels[index] != entity_types['O']:
                                index += 1
                            if index == len(curr_sent_ids):
                                break
                            else:
                                max_sent_length = max(max_sent_length, index)
                                ids.append(curr_sent_ids[:index])
                                labels.append(curr_sent_labels[:index])
                                curr_sent_ids = curr_sent_ids[index:]
                                curr_sent_labels = curr_sent_labels[index:]
                                sentences.append(sentences[-1][index:])
                                sentences[-2] = sentences[-2][:index]
                    ids.append(curr_sent_ids)
                    labels.append(curr_sent_labels)
                    assert len(curr_sent_ids) == len(curr_sent_labels)
                    max_sent_length = max(max_sent_length, len(curr_sent_ids))
                    curr_sent_ids = []
                    curr_sent_labels = []
                    sentences.append([])
                else:
                    doc_start = False
            elif parts[0] == "-DOCSTART-":
                doc_start = True
            else:
                token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(parts[0]))
                curr_sent_labels.append(entity_types[parts[-1]])
                for i in token_ids:
                    curr_sent_ids.append(i)
                for _ in range(len(token_ids)-1):
                    curr_sent_labels.append(entity_types['X'])
                sentences[-1].append(parts[0])
        ids_arr = np.zeros((len(ids), max_sent_length))
        labels_arr = np.zeros((len(ids), max_sent_length, len(entity_types)))
        ids_mask = np.zeros((len(ids), max_sent_length))
        x_indices = np.ones((len(ids), max_sent_length))
        for i, l in enumerate(ids):
            ids_arr[i,:len(l)] = l
            for j in range(len(l)):
                labels_arr[i,j,labels[i][j]] = 1
                if labels[i][j] == entity_types['X']:
                    x_indices[i][labels[i][j]] = 0
            ids_mask[i,:len(l)] = 1
            x_indices[i,len(l):] = 0
        print(max_sent_length)
        return ids_arr, labels_arr, ids_mask, x_indices, sentences

    def get_all_entity_tuples(self, labels_arr_float, x_indices):
        labels_arr = np.argmax(labels_arr_float.astype(np.int32), axis=2)
        entities = set()
        offset = 0
        for i in range(len(labels_arr)):
            curr_start = -1
            curr_type = None
            if np.sum(x_indices[i]) == 0:
                continue
            end_index = len(x_indices[i])-1-list(x_indices[i])[::-1].index(1)
            for j in range(end_index+1):
                # print(entity_types_list[labels_arr[i][j]])
                if entity_types_list[labels_arr[i][j]][0] == 'B':
                    entities.add((curr_type, curr_start, offset+j-1))
                    curr_start = offset+j
                    curr_type = entity_types_list[labels_arr[i][j]][2:]
                elif entity_types_list[labels_arr[i][j]][0] == 'I':
                    if curr_start == -1:
                        curr_start = offset+j
                        curr_type = entity_types_list[labels_arr[i][j]][2:]
                elif entity_types_list[labels_arr[i][j]][0] == 'O':
                    if curr_start > -1:
                        entities.add((curr_type, curr_start, offset+j-1))
                        curr_start = -1
                        curr_type = None
            if curr_start > -1:
                entities.add((curr_type, curr_start, offset+end_index))
            offset += end_index+1
        return entities

    def cache_eval_data(self, eval_file='eng.testa'):
        self.dev_ids_arr, self.dev_labels_arr, self.dev_ids_mask, self.dev_x_indices, self.dev_sentences = self.process_conll_data(os.path.join(conll2003_root, eval_file), is_training=False)
        self.gold_entities = self.get_all_entity_tuples(self.dev_labels_arr, self.dev_x_indices)

    def eval(self, sess=None):
        preds = None
        predicted_entities = set()
        batch_size = 20
        full_preds = np.zeros((self.dev_ids_arr.shape[0], self.dev_ids_arr.shape[1], len(entity_types)))
        for start in range(0, self.dev_ids_arr.shape[0], batch_size):
            end = min(self.dev_ids_arr.shape[0], start+batch_size)
            if sess is None:
                with tf.Session() as sess:
                    preds = sess.run(self.predictions, feed_dict={self.input_ids: self.dev_ids_arr[start:end,:], self.input_mask: self.dev_ids_mask[start:end,:], self.is_training: False})
            else:
                preds = sess.run(self.predictions, feed_dict={self.input_ids: self.dev_ids_arr[start:end,:], self.input_mask: self.dev_ids_mask[start:end,:], self.is_training: False})
            # predicted_entities = predicted_entities.union(self.get_all_entity_tuples(preds, self.dev_x_indices[start:end,:]))
            full_preds[start:end,:] = preds
        predicted_entities = self.get_all_entity_tuples(full_preds, self.dev_x_indices)
        correct_entities = predicted_entities.intersection(self.gold_entities)
        print(len(predicted_entities), len(self.gold_entities))
        precision = 0.0
        recall = 0.0
        if len(predicted_entities) == 0:
            precision = 1.0
            recall = 0.0
        elif len(self.gold_entities) == 0:
            precision = 0.0
            recall = 1.0
        else:
            precision = float(len(correct_entities))/len(predicted_entities)
            recall = float(len(correct_entities))/len(self.gold_entities)
        f1 = 2*precision*recall/(precision+recall)
        return f1

    def train(self):
        train_ids_arr, labels_arr, ids_mask, x_indices, train_sentences = self.process_conll_data(os.path.join(conll2003_root, 'eng.train'))
        x_indicator = tf.placeholder(tf.float32, [None, None])
        o_count = 0
        labels_arr_2d = np.argmax(labels_arr, axis=2)
        for i in range(len(labels_arr_2d)):
            for j in range(len(labels_arr_2d[i])):
                if entity_types_list[labels_arr_2d[i][j]] == 'O' and x_indices[i][j] == 1:
                    o_count += 1
        sys.stdout.flush()
        # loss = tf.reduce_sum(tf.multiply(x_indicator, tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.predictions)))
        loss = tf.reduce_sum(-tf.multiply(x_indicator, tf.reduce_sum(tf.multiply(self.Y, self.predictions), -1)))
        learning_rate = 5e-5
        num_epochs = 50
        batch_size = 64
        num_train_steps = int(num_epochs*len(train_ids_arr)/batch_size)
        num_warmup_steps = int(num_train_steps*0.1)
        train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
        # train_op = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss)
        bert_ckpt_names = [pair[0] for pair in checkpoint_utils.list_variables(bert_checkpoint)]
        bert_vars = [v for v in tf.trainable_variables() if v.name.split(':')[0] in bert_ckpt_names]
        print(len(bert_vars))
        bert_saver = tf.train.Saver(bert_vars)
        # saver = tf.train.Saver()
        with tf.Session() as sess:
            epoch = 0
            sess.run(tf.global_variables_initializer())
            bert_saver.restore(sess, bert_checkpoint)
            self.cache_eval_data()
            max_f1 = 0.0
            while epoch < num_epochs:
                total_loss = 0
                for start in range(0, len(train_ids_arr), batch_size):
                    end = min(len(train_ids_arr), start+batch_size)
                    loss_val, _ = sess.run([loss, train_op], feed_dict={self.input_ids: train_ids_arr[start:end,:], self.Y: labels_arr[start:end,:,:], self.input_mask: ids_mask[start:end,:], x_indicator: x_indices[start:end,:], self.is_training: True})
                    total_loss += loss_val
                if epoch % 1 == 0:
                    dev_f1 = self.eval(sess=sess)
                    print('Epoch ' + str(epoch) + ', Loss: ' + str(total_loss) + ', Dev F1: ' + str(dev_f1))
                    sys.stdout.flush()
                    if dev_f1 > max_f1:
                        self.saver.save(sess, self.model_path)
                        max_f1 = dev_f1
                epoch += 1

if __name__ == '__main__':
    model = NerModel()
    model.train()
    with tf.Session() as sess:
        model.restore_model(sess)
        model.cache_eval_data(eval_file="eng.testb")
        print('Test F1: ' + str(model.eval(sess)))

import tensorflow as tf
from pathlib import Path
import re
import numpy as np
from timeit import default_timer as timer
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from typing import List


def time_str():
    return datetime.now().replace(microsecond=0).isoformat().replace(':', '-')


def make_texts_labels(path, label, params):
    files = path.iterdir()
    for file in files:
        try:
            params['texts'].append(file.read_text())
            params['labels'].append(label)
        except:
            print(file)


def make_dict(params: dict, d: str = 'dict_name', texts_name='texts'):
    for text in params[texts_name]:
        t = re.sub(r'<br />', ' ', text)
        t = re.sub(r'[^\w\']', ' ', t)
        t = re.sub(r"( '|' |'$|^')", " ", t)
        t = t.lower()
        text = t.split()
        token_prev, token_prev_prev = '', ''
        for ind, token in enumerate(text):
            if ind > 0:  # bigram
                params[d][' '.join([token_prev, token])] += 1
            if ind > 1:  # trigram
                params[d][' '.join([token_prev_prev, token_prev, token])] += 1
            params[d][token] += 1
            token_prev_prev = token_prev
            token_prev = token


def tokenize(params, texts, tok_to_ind, d='dict'):
    doc_to_ngram = []
    set_tokens = set(params[d].keys())
    for text in texts:
        doc_dict = defaultdict(int)
        t = re.sub(r'<br />', ' ', text)
        t = re.sub(r'[^\w\']', ' ', t)
        t = re.sub(r"( '|' |'$|^')", " ", t)
        t = t.lower()
        text = t.split()
        token_prev, token_prev_prev = '', ''
        for ind, token in enumerate(text):
            if ind > 0:  # bigram
                bigram = ' '.join([token_prev, token])
                if bigram in set_tokens:
                    doc_dict[tok_to_ind[bigram]] += 1
            if ind > 1:  # trigram
                trigram = ' '.join([token_prev_prev, token_prev, token])
                if trigram in set_tokens:
                    doc_dict[tok_to_ind[trigram]] += 1
            if token in set_tokens:
                doc_dict[tok_to_ind[token]] += 1
            token_prev_prev = token_prev
            token_prev = token
        doc_to_ngram.append(doc_dict)
    return doc_to_ngram


def dict_tokens_to_doc(doc_dict, ind_to_tok):
    return ' '.join([ind_to_tok[k] for k, v in doc_dict.items() if len(ind_to_tok[k].split()) == 1])


def gen_batch_2(docs, ngrams, num_doc, norm_probabs, k=5, batch_size=32):
    num_steps = num_doc // batch_size
    print('number of steps: ', num_steps)
    p = np.random.permutation(num_doc)
    perm_docs = docs[p]  # shuffle datalen(perm_ngrams)
    perm_ngrams = ngrams[p]  # unison

    words_inds = np.arange(len(norm_probabs))
    k_neg_batches = np.random.choice(a=words_inds,
                                     size=[num_doc, k],
                                     p=norm_probabs)

    zeros = np.zeros(shape=[batch_size])
    ones = np.ones(shape=[batch_size])
    for i in range(num_steps):
        doc_batch = perm_docs[i * batch_size: (i + 1) * batch_size]
        for j in range(k + 1):
            if j == 0:
                yield doc_batch, perm_ngrams[i * batch_size: (i + 1) * batch_size], ones
            else:
                yield doc_batch, k_neg_batches[i * batch_size: (i + 1) * batch_size, k - 1], zeros


def gen_batch(docs, ngrams, num_doc, norm_probabs, k=5, batch_size=32):
    num_steps = num_doc // batch_size
    # num_batches = num_doc // batch_size
    p = np.random.permutation(num_doc)
    perm_docs = docs[p]  # shuffle data
    perm_ngrams = ngrams[p]  # unison

    reduced_len = (num_doc // batch_size) * batch_size

    perm_ngrams = perm_ngrams[:reduced_len]
    perm_ngrams = np.reshape(perm_ngrams, newshape=[-1, batch_size])

    perm_docs = perm_docs[:reduced_len]
    perm_docs = np.reshape(perm_docs, [-1, batch_size], order='C')
    tmp = np.repeat(perm_docs, repeats=k + 1, axis=0)
    perm_docs = np.reshape(tmp, [-1, batch_size])

    words_inds = np.arange(len(norm_probabs))
    all_ngrams_batches = np.random.choice(a=words_inds,
                                          size=[(k + 1) * reduced_len // batch_size, batch_size],
                                          p=norm_probabs)

    all_ngrams_batches[::k + 1, :] = perm_ngrams
    zeros = np.zeros(shape=[1, batch_size], dtype=np.float32)
    ones = np.ones(shape=[1, batch_size], dtype=np.float32)
    labels = np.tile(np.vstack([ones, np.repeat(zeros, k, axis=0)]), reps=(reduced_len // batch_size, 1))

    assert perm_docs.shape == all_ngrams_batches.shape
    assert all_ngrams_batches.shape == labels.shape
    print(all_ngrams_batches.shape)
    print(num_steps * (k + 1))

    for docs_b, ngram_b, label_b in zip(perm_docs, all_ngrams_batches, labels):
        yield docs_b, ngram_b, label_b


def get_doc_ngram_indexes_from_data(params, texts):
    """
    :param params: dictionary with all the data
    :return: array with document index and ngram index. each pair is from D.
    """
    doc2all_ngrams = tokenize(params, texts=texts, tok_to_ind=params['tok_to_ind'])

    doc2ngram = [(ind, v) for ind, doc in enumerate(doc2all_ngrams) for v, k in doc.items()]

    d, n = zip(*doc2ngram)
    docs = np.array(d)
    ngrams = np.array(n)
    num_tokens = docs.shape[0]

    data_dict = {"docs": docs, 'ngrams': ngrams, 'num_tokens': num_tokens}

    return data_dict


def create_data(texts=None, dict_name='train_texts'):
    params = {}
    if texts is None:
        path_acli = Path.cwd() / 'aclImdb'
        path_acli.exists()

        params['train_labels'] = []
        params[dict_name] = []

        path_train = path_acli / 'train'

        path_train_neg = path_train / 'neg'
        path_train_pos = path_train / 'pos'
        path_train_unsup = path_train / 'unsup'

        path_test = path_acli / 'test'

        path_test_neg = path_test / 'neg'
        path_test_pos = path_test / 'pos'

        make_texts_labels(path_train_pos, label=1, params=params)
        make_texts_labels(path_train_neg, label=0, params=params)
        make_texts_labels(path_test_pos, label=1, params=params)
        make_texts_labels(path_test_neg, label=0, params=params)
        make_texts_labels(path_train_unsup, label=-1, params=params)
    else:
        params[dict_name] = texts

    d = 'dict'
    params[d] = defaultdict(int)

    make_dict(params, d, texts_name=dict_name)
    print(len(params['dict']), "words in texts found")

    cut_dict = {key: val for key, val in params[d].items() if val > 4}
    print('after cut: ', len(cut_dict))

    params[d] = cut_dict

    probabs = np.array(list(params[d].values())) ** (3 / 4)
    norm_probabs = probabs / np.sum(probabs)

    params['tok_to_ind'] = {k: ind for ind, (k, v) in enumerate(cut_dict.items())}
    params['ind_to_tok'] = {v: k for k, v in params['tok_to_ind'].items()}

    print(len(params['ind_to_tok']))

    data_dict = get_doc_ngram_indexes_from_data(params, texts=params[dict_name])
    # num_tokens -- it is a number of all words in dataset really!

    return params, data_dict, norm_probabs


# make so that I can share wordembs among all graphs.
class Graph:
    num_instances = 0

    def __init__(self, voc_size, emb_size, doc_num):
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.doc_num = doc_num
        self._create_placeholders()
        self._create_emb_layer()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
        self.__class__.num_instances += 1

    def _create_placeholders(self):
        with tf.name_scope('data'):
            self.ph_docs = tf.placeholder(dtype=tf.int64, shape=None, name='docs_indexes')
            self.ph_ngram = tf.placeholder(dtype=tf.int64, shape=None, name='ngrams_indexes')
            self.ph_ngram_in_d = tf.placeholder(dtype=tf.float32, shape=None, name='ngram_in_doc')
            self.ph_lr = tf.placeholder(dtype=tf.float32)

    def _create_emb_layer(self):
        with tf.name_scope('embed'):
            reuse_vars = False
            if self.__class__.num_instances != 0:
                reuse_vars = True
            with tf.variable_scope('word_emb', reuse=reuse_vars):
                self.word_emb = tf.get_variable(name='word_embs',
                                                shape=[self.voc_size, self.emb_size],
                                                initializer=tf.random_uniform_initializer(-1/self.emb_size, 1/self.emb_size))
            # if reuse_vars:  # for restoring model!!!
            #     self.doc_emb = tf.get_variable(
            #         shape=[self.doc_num, self.emb_size],
            #         initializer=tf.truncated_normal_initializer(stddev=1.0 / (self.emb_size ** 0.5)),
            #         name='doc_embs'
            #     )
            # else:
            self.doc_emb = tf.Variable(
                tf.random_uniform(shape=[self.doc_num, self.emb_size],
                                  minval=-1/self.emb_size,
                                  maxval=1/self.emb_size),
                name='doc_embs'
            )

        print(self.doc_emb.name, self.word_emb.name)

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.vec_ngrams = tf.nn.embedding_lookup(params=self.word_emb,
                                                     ids=self.ph_ngram,
                                                     name='emb_lookup_ngrams')

            self.vec_docs = tf.nn.embedding_lookup(params=self.doc_emb,
                                                   ids=self.ph_docs,
                                                   name='emb_lookup_docs')

            self.docs_dot_ngrams = tf.reduce_sum(
                tf.multiply(self.vec_ngrams, self.vec_docs),
                axis=1,
                name='scalar_product'
            )

            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.docs_dot_ngrams,
                                                                labels=self.ph_ngram_in_d,
                                                                name='loss')

            self.loss_sc = tf.reduce_sum(self.loss,
                                         name='one_num_loss')

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.ph_lr,
                                                               name='grad_descent_all').minimize(self.loss_sc)

            self.optimize_only_doc_emb = tf.train.GradientDescentOptimizer(learning_rate=self.ph_lr,
                                                               			   name='grad_descent_doc').minimize(self.loss_sc, var_list=[self.doc_emb])


    def _create_summary(self):
        with tf.name_scope('summary'):
            self.summ_loss = tf.summary.scalar('loss', self.loss_sc)
            # self.merged = tf.summary.merge_all()


# TODO: need to add feature to load model
class DvNgramModel:
    def __init__(self, batch_size, emb_size, k, lr):
        self.graphs = []
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.k = k
        self.lr = lr

    def _build_model(self, texts=None):
        self._create_train_data(texts)
        self.train_graph = Graph(doc_num=self.train_doc_num, voc_size=self.voc_size, emb_size=self.emb_size)

    def _create_train_data(self, texts=None):
        self.params, self.train_data_dict, self.norm_probabs = create_data(texts,
                                                                           dict_name='train_texts')
        self.voc_size = self.norm_probabs.shape[0]
        self.train_doc_num = len(texts)
        print(self.train_doc_num, self.voc_size)
        self.params['text2id'] = {text: ind for ind, text in enumerate(self.params['train_texts'])}
        print(len(self.params['text2id']))

    @staticmethod
    def load_tensors_from_model(
            path_to_graph='./dvngram_model_lr_0_1/model_nt/model-7.meta',
            path_to_checkpoint='./dvngram_model_lr_0_1/model_nt/model-1'):
        sess = tf.InteractiveSession()

        saver = tf.train.import_meta_graph(path_to_graph)
        saver.restore(sess, path_to_checkpoint)

        graph = tf.get_default_graph()

        d_e = graph.get_tensor_by_name('doc_embs:0')
        n_e = graph.get_tensor_by_name('word_embs:0')

        word_ems = n_e.eval()
        doc_embs = d_e.eval()

        sess.close()

        return doc_embs, word_ems

    def train_log_reg(self, train, labels):
        text2id = self.params['text2id']
        indecies = [text2id[text] for text in train]
        x = self.train_mat[indecies, :]

        # sc = MinMaxScaler()
        # sc.fit(X)
        # train = sc.transform(train)
        # test = sc.transform(test)
        self.log_reg = LogisticRegressionCV(Cs=10,
                                            cv=5,
                                            tol=0.00001,
                                            solver='liblinear',
                                            refit=True,
                                            verbose=0,
                                            max_iter=500,
                                            penalty='l2')

        self.log_reg.fit(X=x, y=labels)

    def predict(self, data: List[str], model_name: str = None) -> np.array:
        text2id = self.params['text2id']
        if data[0] in text2id.keys():
            indecies = [text2id[text] for text in data]
            x = self.train_mat[indecies, :]
        else:
            x = self.train_doc2vec(data, model_name=model_name)
        predicts = self.log_reg.predict(x)
        return predicts

    def train_doc2vec(self, data: List[str], model_name: str):
        data_dict = get_doc_ngram_indexes_from_data(self.params,
                                                    texts=data)

        doc_num = len(data)

        # TODO: how to share word embeddings!!
        self.graphs.append(
            Graph(doc_num=doc_num,
                  voc_size=self.voc_size,
                  emb_size=self.emb_size)
        )

        if model_name is None:
            model_name = 'dvngram_model_cls__' + str(len(self.graphs))

        # TODO: think about epochs
        # TODO: check word_emb_mat!!!
        # self.num_epochs
        if doc_num > 20_000:
            num_epochs = 10
        if doc_num < 15_000:
            num_epochs = 15
            self.lr = 0.001
        # num_epochs = 3_000_000 // ((data_dict['num_tokens'] // self.batch_size) * (self.k + 1)) + 1
        # print((data_dict['num_tokens'] // self.batch_size), ((data_dict['num_tokens'] // self.batch_size) * (self.k + 1)))
        # # num_epochs = max(2, round(doc_num / self.train_doc_num * 5))
        # if num_epochs > 100:
        #     num_epochs = 5
        #     self.lr = self.lr / 5

        print(num_epochs, doc_num)
        # num_epochs = 0
        word_emb_mat, doc_emb_mat = \
            self.run_graph(graph=self.graphs[-1],
                           num_epochs=num_epochs,
                           data_dict=data_dict,
                           model_name=model_name,
                           use_frozen_doc_emb=True)

        # TODO: update text2id!!! find on internet how to concat two dicts.
        # self.params['text2id'] = {text: ind for ind, text in enumerate(self.params['train_texts'])}

        return doc_emb_mat

    # TODO: here is going to happen training of new doc2vecs.
    def train(self, train_texts, num_epochs=5, model_name='dvngram_model_train'):
        self._build_model(texts=train_texts)

        self.word_emb_mat, self.train_mat = self.run_graph(graph=self.train_graph,
                                                           num_epochs=num_epochs,
                                                           data_dict=self.train_data_dict,
                                                           model_name=model_name,
                                                           try_restore_model=False)

    def run_graph(self,
                  graph,
                  num_epochs,
                  data_dict,
                  model_name='./dvngram_model_4',
                  try_restore_model=False,
                  use_frozen_doc_emb=False):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver(max_to_keep=1)
        start = timer()

        docs = data_dict['docs']
        ngrams = data_dict['ngrams']
        num_tokens = data_dict['num_tokens']

        num_batches = num_tokens // self.batch_size

        new_graph = tf.get_default_graph()
        # optim = graph.optimizer

        if try_restore_model:
            new_graph = tf.Graph()

        with tf.Session(config=config, graph=new_graph) as sess:
            already_epoch = 0
            if try_restore_model:
                found = ''
                path_to_graphs = Path.cwd() / model_name / 'model_nt'
                for gr in path_to_graphs.iterdir():
                    if 'meta' in gr.name:
                        found = gr
                        cand_epoch = int(re.findall(r'\d+', found.name)[0])
                        if cand_epoch > already_epoch:
                            already_epoch = cand_epoch
                saver = tf.train.import_meta_graph(str(found))
                saver.restore(sess, tf.train.latest_checkpoint(str(path_to_graphs)))
                # graph = tf.get_default_graph()
                # print('opt: ', optimizer)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(model_name + '/summaries%s' % time_str(), sess.graph)
            step = 0
            lr = self.lr
            is_lr_cut = True  # don't use a lot of this 
            thresh = round(num_epochs * 0.7)
            print(thresh)
            for i in range(num_epochs):
                total_loss = 0
                if i > thresh and not is_lr_cut:
                    lr = lr / 2
                    is_lr_cut = True

                if use_frozen_doc_emb and i < num_epochs - 5:
                    optim = graph.optimize_only_doc_emb
                else:
                    optim = graph.optimizer

                with tqdm(total=num_batches * (self.k + 1)) as p:
                    batches = gen_batch(docs,
                                        ngrams,
                                        num_tokens,
                                        self.norm_probabs,
                                        k=self.k,
                                        batch_size=self.batch_size)

                    for ph1, ph2, ph3 in batches:
                        _ = sess.run(
                            optim,
                            feed_dict={
                                graph.ph_lr: lr,
                                graph.ph_docs: ph1,
                                graph.ph_ngram: ph2,
                                graph.ph_ngram_in_d: ph3
                            }
                        )
                        p.update(1)
                        if step % 5_000 == 0:
                            summary, l = sess.run(
                                [graph.summ_loss, graph.loss_sc],
                                feed_dict={
                                    graph.ph_lr: lr,
                                    graph.ph_docs: ph1,
                                    graph.ph_ngram: ph2,
                                    graph.ph_ngram_in_d: ph3
                                }
                            )
                            total_loss += l
                            train_writer.add_summary(summary, global_step=step)
                        step += 1

                if (i + 1) % 1 == 0:
                    print('Epoch {0}: {1}'.format(i, total_loss))
                    saver.save(sess,
                               model_name + '/model_nt/model',
                               global_step=i + already_epoch)
            try:
                word_emb_mat = graph.word_emb.eval(sess)
                doc_emb_mat = graph.doc_emb.eval(sess)
            except:
                print('tried')
            print(word_emb_mat.shape)
        end = timer()
        print('Took: %f seconds' % (end - start))
        return word_emb_mat, doc_emb_mat


def pretrain(texts):
    # got all texts
    params = {'texts': texts, 'labels': [-1 for _ in range(len(texts))]}
    n_model = DvNgramModel(batch_size=100,
                           emb_size=500,
                           k=5,
                           lr=0.05)

    n_model.train(num_epochs=8, train_texts=params['texts'], model_name='dvngram_model_train')
    return n_model


def train(texts: List[str], labels: List[float], n_model=None):
    n_model.train_log_reg(train=texts, labels=labels)
    return n_model


def classify(texts: List[str], n_model: DvNgramModel):
    pred_labels = n_model.predict(texts)
    return pred_labels


# make different
def _visualize():
    pass
    # most_popular_words = sorted(params[d].items(), key=lambda x: -x[1])[0:100_000]
    #
    # indexes = np.array([tok_to_ind[k] for k, v in most_popular_words])
    #
    # indexes.shape
    #
    # mod_word_embs = word_ems / np.sum(word_ems ** 2, axis=1)[:, np.newaxis]
    #
    # with open('./visualize/most_popular.tsv', 'w') as f:
    #     l = [str(k) + '\t' + str(v) for k, v in list(most_popular_words)[0:100_000]]
    #     f.write('word\tfreq\n')
    #     f.write('\n'.join(l))
    #
    # from tensorflow.contrib.tensorboard.plugins import projector
    #
    # mod_word_embs[indexes].shape
    #
    # embedding_var = tf.Variable(mod_word_embs[indexes], name='embeded_words')
    # sess.run(embedding_var.initializer)
    #
    # config = projector.ProjectorConfig()
    # summary_writer = tf.summary.FileWriter('./visualize')
    #
    # embedding = config.embeddings.add()
    #
    # embedding.tensor_name = embedding_var.name
    #
    # embedding
    #
    # embedding.metadata_path = 'most_popular.tsv'
    #
    # projector.visualize_embeddings(summary_writer, config)
    #
    # saver_embed = tf.train.Saver([embedding_var])
    # saver_embed.save(sess, './visualize/model.ckpt', 1)


def main():
    path_acli = Path('/Users/kononykhindanil/Downloads/aclImdb_clean')
    print(path_acli.exists())

    train_params = {'texts': [], 'labels': []}
    pretrain_params = {'texts': [], 'labels': []}
    test_params = {'texts': [], 'labels': []}

    path_train = path_acli / 'train'

    path_train_neg = path_train / 'neg'
    path_train_pos = path_train / 'pos'
    path_train_unsup = path_train / 'unsup'

    path_test = path_acli / 'test'

    path_test_neg = path_test / 'neg'
    path_test_pos = path_test / 'pos'

    make_texts_labels(path_train_pos, label=1, params=train_params)
    make_texts_labels(path_train_neg, label=0, params=train_params)
    make_texts_labels(path_test_pos, label=1, params=test_params)
    make_texts_labels(path_test_neg, label=0, params=test_params)
    make_texts_labels(path_train_unsup, label=-1, params=pretrain_params)

    par = pretrain(texts=pretrain_params['texts'][0:200])
    tt = train_params['texts']
    tl = train_params['labels']
    par2 = train(texts=tt, labels=tl, n_model=par)
    pred = classify(texts=tt, n_model=par2)
    k = 0
    for i, _ in enumerate(pred):
        if tl[i] == pred[i]:
            k += 1
    print()
    print(k / len(pred))
    print()
    tt = test_params['texts']
    tl = test_params['labels']
    pred = classify(texts=tt, n_model=par2)
    k = 0
    for i, _ in enumerate(pred):
        if tl[i] == pred[i]:
            k += 1
    print()
    print(k / len(pred))
    print()


# main()

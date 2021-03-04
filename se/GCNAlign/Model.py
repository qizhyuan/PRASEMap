import time
import KGs
import math
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import scipy.sparse as sp
import numpy as np
from scipy.spatial.distance import cdist
tf.disable_v2_behavior()

_LAYER_UIDS = {}


def construct_feed_dict(features, support, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict


def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def glorot(shape, name=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def trunc_normal(shape, name=None, normalize=True):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(shape[0])))
    if not normalize:
        return initial
    return tf.nn.l2_normalize(initial, 1)


def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name + ":0")
    except:
        return tf.placeholder(tf.int32, name=name)


def load_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def align_loss(outlayer, ILL, gamma, k):
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)

    neg_left = get_placeholder_by_name("neg_left")
    neg_right = get_placeholder_by_name("neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma

    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

    neg_left = get_placeholder_by_name("neg2_left")
    neg_right = get_placeholder_by_name("neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCNAlignUnit(Model):
    def __init__(self, placeholders, input_dim, output_dim, ILL, lr=1e-1, sparse_inputs=False, featureless=True,
                 neg_num=5, gamma=3):
        super(GCNAlignUnit, self).__init__()

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ILL = ILL
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.lr = lr
        self.neg_num = neg_num
        self.gamma = gamma

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.build()

    def _loss(self):
        self.loss += align_loss(self.outputs, self.ILL, self.gamma, self.neg_num)

    def _accuracy(self):
        pass

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            transform=False,
                                            init=trunc_normal))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            transform=False))


class GraphConvolution(Layer):

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, transform=True, init=glorot):
        super(GraphConvolution, self).__init__()

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transform = transform

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if input_dim == output_dim and not self.transform and not featureless:
                    continue
                self.vars['weights_' + str(i)] = init([input_dim, output_dim], name=('weights_' + str(i)))

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        if self.dropout:
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)

        supports = list()
        for i in range(len(self.support)):
            if 'weights_' + str(i) in self.vars:
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
            else:
                pre_sup = x
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GCNAlign:

    def __init__(self, kgs: KGs, embed_dim=64, layers_dims=None, activations=None, dropout=0., lr=8, neg_num=5,
                 epoch_num=100, margin=3):
        self.layers_dims = layers_dims
        self.activations = activations
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.kgs = kgs
        self.lr = lr
        self.margin = margin
        self.neg_num = neg_num
        self.epoch_num = epoch_num
        self.support_number = 1

        self.ae_input = None
        self.support = None
        self.adj = None
        self.ph_ae = None
        self.ph_se = None
        self.model_ae = None
        self.model_se = None
        self.feed_dict_se = None
        self.feed_dict_ae = None

        self.gcn_se = None
        self.gcn_ae = None

        self.se_input_dim = None
        self.ae_input_dim = None

        self.embed_idx_dict = dict()
        self.embed_idx_dict_inv = dict()

        self.ent_training_links = list()

        self.kg1_ent_emb_id_list = list()
        self.kg2_ent_emb_id_list = list()

        self.kg1_train_ent_lite_list = list()
        self.kg2_train_ent_lite_list = list()

        self.kg1_test_ent_list = list()
        self.kg2_test_ent_list = list()

        self.session = None

        self.vec_se = None
        self.vec_ae = None

    def _reindex(self):
        embed_idx = 0
        self.embed_idx_dict.clear()
        self.embed_idx_dict_inv.clear()
        for item in self.kgs.kg1.get_ent_id_set() | self.kgs.kg2.get_ent_id_set():
            if not self.embed_idx_dict.__contains__(item):
                self.embed_idx_dict[item] = embed_idx
                self.embed_idx_dict_inv[embed_idx] = item
                embed_idx += 1

        for (lite_id, lite_name) in self.kgs.kg1.lite_id_name_dict.items():
            if not self.embed_idx_dict.__contains__(lite_id):
                self.embed_idx_dict[lite_id] = embed_idx
                self.embed_idx_dict_inv[embed_idx] = lite_id
                embed_idx += 1

        for (lite_id, lite_name) in self.kgs.kg2.lite_id_name_dict.items():
            if not self.embed_idx_dict.__contains__(lite_id):
                if self.kgs.kg1.name_lite_id_dict.__contains__(lite_name):
                    lite_cp_id = self.kgs.kg1.name_lite_id_dict[lite_name]
                    lite_cp_embed_id = self.embed_idx_dict[lite_cp_id]
                    self.embed_idx_dict[lite_id] = lite_cp_embed_id
                else:
                    self.embed_idx_dict[lite_id] = embed_idx
                    self.embed_idx_dict_inv[embed_idx] = lite_id
                    embed_idx += 1

    def _load_attr(self):
        cnt = {}
        for (h, r, t) in self.kgs.kg1.get_attribute_id_triples():
            if h not in self.kgs.kg1.get_ent_id_set():
                continue
            if r not in cnt:
                cnt[r] = 1
            else:
                cnt[r] += 1

        for (h, r, t) in self.kgs.kg2.get_attribute_id_triples():
            if h not in self.kgs.kg2.get_ent_id_set():
                continue
            if r not in cnt:
                cnt[r] = 1
            else:
                cnt[r] += 1

        fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
        print(fre)
        attr2id = {}
        num = int(0.7 * len(cnt))
        for i in range(num):
            attr2id[fre[i][0]] = i

        attr = np.zeros((self.kgs.get_entity_nums(), num), dtype=np.float32)
        for (h, r, t) in self.kgs.kg1.get_attribute_id_triples() | self.kgs.kg2.get_attribute_id_triples():
            if r in attr2id:
                attr[self.embed_idx_dict[h]][attr2id[r]] = 1.0
        self.attr = attr

    def _init_weight_adj(self):
        weight_dict = dict()
        for (h, r, t) in self.kgs.kg1.get_relation_id_triples() | self.kgs.kg2.get_relation_id_triples():
            h_emb_id, t_emb_id = self.embed_idx_dict[h], self.embed_idx_dict[t]
            if h_emb_id == t_emb_id:
                continue
            else:
                inv_functionality = max(self.kgs.kg1.get_inv_functionality_by_id(t_emb_id),
                                        self.kgs.kg2.get_inv_functionality_by_id(t_emb_id))
                if (h_emb_id, t_emb_id) not in weight_dict:
                    weight_dict[(h_emb_id, t_emb_id)] = max(inv_functionality, 0.3)
                else:
                    weight_dict[(h_emb_id, t_emb_id)] += max(inv_functionality, 0.3)
        row, col, data = [], [], []
        for (key, value) in weight_dict.items():
            row.append(key[0])
            col.append(key[1])
            data.append(value)

        self.adj = sp.coo_matrix((data, (row, col)), shape=(self.kgs.get_entity_nums(), self.kgs.get_entity_nums()))

    def _load_data(self):
        self._init_weight_adj()
        self.ae_input = sparse_to_tuple(sp.coo_matrix(self.attr))
        self.se_input_dim = self.ae_input[2][0]
        self.ae_input_dim = self.ae_input[2][1]
        self.support = [preprocess_adj(self.adj)]
        self._init_train_data()
        return

    def _init_train_data(self, threshold=0.1):
        self.ent_training_links.clear()
        self.kg1_test_ent_list.clear()
        self.kg2_test_ent_list.clear()
        for (e1, e2, p) in self.kgs.get_ent_align_ids_result():
            if p < threshold:
                continue
            idx1, idx2 = self.embed_idx_dict[e1], self.embed_idx_dict[e2]
            self.ent_training_links.append([idx1, idx2])

        for ent in self.kgs.kg1.get_ent_id_set():
            self.kg1_ent_emb_id_list.append(self.embed_idx_dict[ent])

        for ent in self.kgs.kg2.get_ent_id_set():
            self.kg2_ent_emb_id_list.append(self.embed_idx_dict[ent])

        for item in self.kgs.get_kg1_unaligned_candidate_ids():
            self.kg1_test_ent_list.append(self.embed_idx_dict[item])
        for item in self.kgs.get_kg2_unaligned_candidate_ids():
            self.kg2_test_ent_list.append(self.embed_idx_dict[item])

        self.train = np.array(self.ent_training_links)

    def _init_model(self):
        self.ph_ae = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(self.support_number)],
            "features": tf.sparse_placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0., shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=())
        }
        self.ph_se = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(self.support_number)],
            "features": tf.placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0., shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=())
        }
        self.model_ae = GCNAlignUnit(self.ph_ae, input_dim=self.ae_input_dim, output_dim=self.embed_dim,
                                     ILL=self.train, sparse_inputs=True, featureless=False, neg_num=self.neg_num,
                                     lr=self.lr, gamma=self.margin)
        self.model_se = GCNAlignUnit(self.ph_se, input_dim=self.se_input_dim, output_dim=self.embed_dim,
                                     ILL=self.train,
                                     sparse_inputs=False, featureless=True, neg_num=self.neg_num, lr=self.lr,
                                     gamma=self.margin)

        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def init(self):
        self._reindex()
        self._load_attr()
        self._load_data()
        self._init_model()

    def train_embeddings(self):
        neg_num = self.neg_num
        train_num = len(self.ent_training_links)
        train_links = np.array(self.ent_training_links)
        pos = np.ones((train_num, neg_num)) * (train_links[:, 0].reshape((train_num, 1)))
        neg_left = pos.reshape((train_num * neg_num,))
        pos = np.ones((train_num, neg_num)) * (train_links[:, 1].reshape((train_num, 1)))
        neg2_right = pos.reshape((train_num * neg_num,))
        neg2_left = None
        neg_right = None
        feed_dict_se = None
        feed_dict_ae = None

        for i in range(1, self.epoch_num + 1):
            start = time.time()
            if i % 10 == 1:
                neg2_left = np.random.choice(self.se_input_dim, train_num * neg_num)
                neg_right = np.random.choice(self.se_input_dim, train_num * neg_num)

            feed_dict_ae = construct_feed_dict(self.ae_input, self.support, self.ph_ae)
            feed_dict_ae.update({self.ph_ae['dropout']: self.dropout})
            feed_dict_ae.update({'neg_left:0': neg_left, 'neg_right:0': neg_right,
                                 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
            feed_dict_se = construct_feed_dict(1., self.support, self.ph_se)
            feed_dict_se.update({self.ph_se['dropout']: self.dropout})
            feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right,
                                 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
            batch_loss1, _ = self.session.run(fetches=[self.model_ae.loss, self.model_ae.opt_op],
                                              feed_dict=feed_dict_ae)
            batch_loss2, _ = self.session.run(fetches=[self.model_se.loss, self.model_se.opt_op],
                                              feed_dict=feed_dict_se)

            batch_loss = batch_loss1 + batch_loss2
            print('epoch {}, average triple loss: {:.4f}, cost time: {:.4f}s'.format(i, batch_loss,
                                                                                     time.time() - start))
            # if i % 10 == 0:
            #     self.feed_dict_ae = feed_dict_ae
            #     self.feed_dict_se = feed_dict_se
            #     self.test()
            #     self.valid(path=r"D:\repos\self\PARIS-PYTHON\dataset\industry\ent_links")

        vec_se = self.session.run(self.model_se.outputs, feed_dict=feed_dict_se)
        vec_ae = self.session.run(self.model_ae.outputs, feed_dict=feed_dict_ae)
        self.vec_se = vec_se
        self.vec_ae = vec_ae
        return vec_se, vec_ae

    def mapping_feed_back_to_pr(self, beta=0.9, prob=0.95):
        embeddings = np.concatenate([self.vec_se * beta, self.vec_ae * (1.0 - beta)], axis=1)
        embeds1 = np.array([embeddings[e] for e in self.kg1_test_ent_list])
        embeds2 = np.array([embeddings[e] for e in self.kg2_test_ent_list])
        distance = cdist(embeds1, embeds2, "cityblock")

        kg1_counterpart = np.argmin(distance, axis=1)
        kg2_counterpart = np.argmin(distance, axis=0)
        kg1_matched_pairs = set([(i, kg1_counterpart[i]) for i in range(len(kg1_counterpart))])
        kg2_matched_pairs = set([(kg2_counterpart[i], i) for i in range(len(kg2_counterpart))])
        kg_matched_pairs = kg1_matched_pairs & kg2_matched_pairs

        for (kg1_ent, kg2_ent) in kg_matched_pairs:
            kg1_emb_id, kg2_emb_id = self.kg1_test_ent_list[kg1_ent], self.kg2_test_ent_list[kg2_ent]
            kg1_id, kg2_id = self.embed_idx_dict_inv[kg1_emb_id], self.embed_idx_dict_inv[kg2_emb_id]
            self.kgs.insert_ent_eqv_both_way_by_id(kg1_id, kg2_id, prob)

    def embedding_feed_back_to_pr(self, beta=0.9):
        embeddings = np.concatenate([self.vec_se * beta, self.vec_ae * (1.0 - beta)], axis=1)
        for ent in self.kgs.kg1.get_ent_id_set():
            ent_emb_id = self.embed_idx_dict[ent]
            self.kgs.kg1.insert_ent_embed_by_id(ent, embeddings[ent_emb_id, :])

        for ent in self.kgs.kg2.get_ent_id_set():
            ent_emb_id = self.embed_idx_dict[ent]
            self.kgs.kg2.insert_ent_embed_by_id(ent, embeddings[ent_emb_id, :])

    def feed_back_to_pr_module(self, mapping_feedback=True, embedding_feedback=True, beta=0.9, prob=0.95):
        if mapping_feedback:
            self.mapping_feed_back_to_pr(beta, prob)
        if embedding_feedback:
            self.embedding_feed_back_to_pr(beta)

    # def test(self, beta=0.9):
    #     vec_se = self.session.run(self.model_se.outputs, feed_dict=self.feed_dict_se)
    #     vec_ae = self.session.run(self.model_ae.outputs, feed_dict=self.feed_dict_ae)
    #     embeddings = np.concatenate([vec_se * beta, vec_ae * (1.0 - beta)], axis=1)
    #     embeds1 = np.array([embeddings[e] for e in self.kg1_ent_emb_id_list])
    #     embeds2 = np.array([embeddings[e] for e in self.kg2_ent_emb_id_list])
    #     distance = cdist(embeds1, embeds2, "cityblock")
    #     kg1_counterpart = np.argmin(distance, axis=1)
    #     kg2_counterpart = np.argmin(distance, axis=0)
    #     kg1_matched_pairs = set([(i, kg1_counterpart[i]) for i in range(len(kg1_counterpart))])
    #
    #     kg2_matched_pairs = set([(kg2_counterpart[i], i) for i in range(len(kg2_counterpart))])
    #     kg_matched_pairs = kg1_matched_pairs & kg2_matched_pairs
    #     kg_matched_set = set()
    #     for (ent, ent_cp) in kg_matched_pairs:
    #         kg_matched_set.add((self.kg1_ent_emb_id_list[ent], self.kg2_ent_emb_id_list[ent_cp]))
    #     train_set = set()
    #     for i in range(len(self.ent_training_links)):
    #         train_set.add((self.ent_training_links[i][0], self.ent_training_links[i][1]))
    #
    #     if len(kg_matched_pairs) > 0:
    #         recall = len(train_set & kg_matched_set) / len(train_set)
    #         precision = len(train_set & kg_matched_set) / len(kg_matched_set)
    #         print("recall:\t" + str(recall) + "\tprecision:\t" + str(precision))
    #
    # def valid(self, beta=0.9, path=None):
    #     vec_se = self.session.run(self.model_se.outputs, feed_dict=self.feed_dict_se)
    #     vec_ae = self.session.run(self.model_ae.outputs, feed_dict=self.feed_dict_ae)
    #     embeddings = np.concatenate([vec_se * beta, vec_ae * (1.0 - beta)], axis=1)
    #     embeds1 = np.array([embeddings[e] for e in self.kg1_test_ent_list])
    #     embeds2 = np.array([embeddings[e] for e in self.kg2_test_ent_list])
    #     distance = cdist(embeds1, embeds2, "cityblock")
    #     kg1_counterpart = np.argmin(distance, axis=1)
    #     kg2_counterpart = np.argmin(distance, axis=0)
    #     kg1_matched_pairs = set([(i, kg1_counterpart[i]) for i in range(len(kg1_counterpart))])
    #
    #     kg2_matched_pairs = set([(kg2_counterpart[i], i) for i in range(len(kg2_counterpart))])
    #     kg_matched_pairs = kg1_matched_pairs & kg2_matched_pairs
    #     kg_matched_set = set()
    #     for (ent, ent_cp) in kg_matched_pairs:
    #         kg_matched_set.add((self.kg1_test_ent_list[ent], self.kg2_test_ent_list[ent_cp]))
    #     valid_set = set()
    #     with open(path, "r", encoding="utf8") as f:
    #         for line in f.readlines():
    #             params = str.strip(line).split("\t")
    #             ent_l, ent_r = params[0].strip(), params[1].strip()
    #             obj_l, obj_r = self.kgs.kg1.get_ent_id_without_insert(ent_l), self.kgs.kg2.get_ent_id_without_insert(
    #                 ent_r)
    #             if obj_l is None:
    #                 print("Exception: fail to load Entity (" + ent_l + ")")
    #             if obj_r is None:
    #                 print("Exception: fail to load Entity (" + ent_r + ")")
    #             if obj_l is None or obj_r is None:
    #                 continue
    #             if obj_l in self.kgs.get_kg1_unaligned_candidates() and obj_r in self.kgs.get_kg2_unaligned_candidates():
    #                 valid_set.add((self.embed_idx_dict[obj_l], self.embed_idx_dict[obj_r]))
    #
    #     if len(kg_matched_pairs) > 0:
    #         recall = len(valid_set & kg_matched_set) / len(valid_set)
    #         precision = len(valid_set & kg_matched_set) / len(kg_matched_set)
    #         print("recall:\t" + str(recall) + "\tprecision:\t" + str(precision))

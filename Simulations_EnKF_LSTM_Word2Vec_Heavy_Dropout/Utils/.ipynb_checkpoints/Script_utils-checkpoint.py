from sklearn.model_selection import train_test_split
lr = 1e-3
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
import tensorflow as tf
import random
import numpy as np 
# os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value



# # 4. Set the `tensorflow` pseudo-random generator at a fixed value


from tqdm.notebook import tqdm

# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# from keras import backend as K
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)
# import numpy as np
# import random as rn

# sd = 1 # Here sd means seed.
# np.random.seed(sd)
# rn.seed(sd)
# os.environ['PYTHONHASHSEED']=str(sd)

# from keras import backend as K
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
# tf.compat.v1.set_random_seed(sd)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
# K.set_session(sess)

# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()
# import numpy as np
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# from tensorflow.keras import backend as K
# K.set_session
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# # K.set_session(sess)


def attention_lstm_model(training, model_cbow): 
    padding_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    unknown_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    weight_vectors = np.vstack((padding_vector, unknown_vector))
    weight_vectors = np.vstack((weight_vectors, model_cbow.wv.vectors))
    embedding_layer = tf.keras.layers.Embedding(len(weight_vectors),
                            weight_vectors.shape[1],
                            weights=[weight_vectors],
                            mask_zero = False,
                            trainable=False)

    
    vectorize_layer = tf.keras.layers.TextVectorization(
                     output_mode='int',
                     vocabulary=model_cbow.wv.index_to_key, 
                     standardize = None)
    
    input_layer = tf.keras.layers.Input(shape = (1,), dtype = tf.string)

    vectorize = vectorize_layer(input_layer)

    vectorize.shape

    emb_output = embedding_layer(vectorize)

    emb_output.shape
    
    # spatial_drop = tf.keras.layers.SpatialDropout1D(0.5)
    
    # spatial_out = spatial_drop(emb_output)

    lstm_layer = tf.keras.layers.LSTM(25, return_sequences = True, dropout = 0.5)

    lstm_output = lstm_layer(emb_output, training = training)

#     x_a = tf.keras.layers.Dense(lstm_output.get_shape()[-1]//2, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp")(lstm_output) 
    
    x_a = tf.keras.layers.Dropout(0.75)(lstm_output, training = training)
    
    x_a = tf.keras.layers.Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context", kernel_regularizer=tf.keras.regularizers.L2())(x_a)

    x_a = tf.keras.layers.Flatten()(x_a)

    att_out = tf.keras.layers.Activation('softmax')(x_a) 

    x_a2 = tf.keras.layers.RepeatVector(lstm_output.get_shape()[-1])(att_out)

    x_a2 = tf.keras.layers.Permute([2,1])(x_a2)

    out = tf.keras.layers.Multiply()([lstm_output,x_a2])
    
    out = tf.keras.layers.Lambda(lambda x : tf.math.reduce_sum(x, axis = 1), name='expectation_over_words')(out)
    
    dropout_layer = tf.keras.layers.Dropout(0.65)(out, training = training)

    pred_head = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2())

    pred_output = pred_head(dropout_layer)

    model = tf.keras.models.Model(input_layer, pred_output)
    
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = lr), 
                 metrics=tf.keras.metrics.BinaryAccuracy())
    
    return model


def simple_lstm(training, model_cbow):
    padding_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    unknown_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    weight_vectors = np.vstack((padding_vector, unknown_vector))
    weight_vectors = np.vstack((weight_vectors, model_cbow.wv.vectors))
    embedding_layer = tf.keras.layers.Embedding(len(weight_vectors),
                            weight_vectors.shape[1],
                            weights=[weight_vectors],
                            mask_zero = False,
                            trainable=False)
    
    vectorize_layer = tf.keras.layers.TextVectorization(
                     output_mode='int',
                     vocabulary=model_cbow.wv.index_to_key, 
                     standardize = None)
    
    input_layer = tf.keras.layers.Input(shape = (1,), dtype = tf.string)

    vectorize = vectorize_layer(input_layer)

    vectorize.shape

    emb_output = embedding_layer(vectorize)

    emb_output.shape

    lstm_layer = tf.keras.layers.LSTM(100,  dropout = 0.6)

    lstm_output = lstm_layer(emb_output, training = training)

    dropout_layer = tf.keras.layers.Dropout(0.6)(lstm_output, training = training)

    pred_head = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2())

    pred_output = pred_head(dropout_layer)

    model = tf.keras.models.Model(input_layer, pred_output)
    
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = lr), 
                 metrics= "accuracy")
    
    return model



def get_data_splits(data_input, features_input, idx):
    
    X_train, X_test, y_train, y_test = train_test_split(features_input, data_input["high_level_substr"], test_size = 0.3,
                                                     shuffle = True, stratify = data_input["high_level_substr"].values)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3,
                                                     shuffle = True, stratify = y_train.values)
    
    train_idx, valid_idx, test_idx = y_train.index, y_valid.index, y_test.index
    
    
    y_train, y_valid, y_test = (y_train.values == "pectin").astype(float), (y_valid.values == "pectin").astype(float), (y_test.values == "pectin").astype(float)
    
    X_train_word2vec, X_valid_word2vec, X_test_word2vec  = data_input.iloc[train_idx,-50:].reset_index(drop = True), data_input.iloc[valid_idx,-50:].reset_index(drop = True), data_input.iloc[test_idx,-50:].reset_index(drop = True)
    
    return X_train, X_valid, X_test,  y_train, y_valid, y_test, X_train_word2vec, X_valid_word2vec, X_test_word2vec


def get_data_splits_mod2(data_input, features_input, idx):
    
    X_train, X_test, y_train, y_test = train_test_split(features_input, data_input["high_level_substr"], test_size = 0.1,
                                                     shuffle = True, stratify = data_input["high_level_substr"].values)
    
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3,
    #                                                  shuffle = True, stratify = y_train.values)
    
    train_idx, test_idx = y_train.index,  y_test.index
    
    
    y_train,  y_test = (y_train.values == "pectin").astype(float),  (y_test.values == "pectin").astype(float)
    
    X_train_word2vec, X_test_word2vec  = data_input.iloc[train_idx,-50:].reset_index(drop = True), data_input.iloc[test_idx,-50:].reset_index(drop = True)
    
    return X_train, X_test,  y_train, y_test, X_train_word2vec, X_test_word2vec


def get_data_splits_mod1(data_input, features_input, idx):
    
    X_train, X_valid, y_train, y_valid = train_test_split(features_input, data_input["high_level_substr"], test_size = 0.3,
                                                     shuffle = True, stratify = data_input["high_level_substr"].values)
    
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3,
    #                                                  shuffle = True, stratify = y_train.values)
    
    train_idx, valid_idx = y_train.index,  y_valid.index
    
    
    y_train,  y_valid = (y_train.values == "pectin").astype(float),  (y_valid.values == "pectin").astype(float)
    
    X_train_word2vec, X_valid_word2vec  = data_input.iloc[train_idx,-50:].reset_index(drop = True), data_input.iloc[valid_idx,-50:].reset_index(drop = True)
    
    return X_train, X_valid,  y_train, y_valid, X_train_word2vec, X_valid_word2vec



def get_data_splits_old_algo(data_input, features_input, idx):
    
    X_train, X_test, y_train, y_test = train_test_split(features_input, data_input["high_level_substr"], test_size = 0.3,
                                                     shuffle = True, stratify = data_input["high_level_substr"].values)
    
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.33,
                                                     shuffle = True, stratify = y_test.values)
    
    train_idx, valid_idx, test_idx = y_train.index, y_valid.index, y_test.index
    
    
    y_train, y_valid, y_test = (y_train.values == "pectin").astype(float), (y_valid.values == "pectin").astype(float), (y_test.values == "pectin").astype(float)
    
    X_train_word2vec, X_valid_word2vec, X_test_word2vec  = data_input.iloc[train_idx,-50:].reset_index(drop = True), data_input.iloc[valid_idx,-50:].reset_index(drop = True), data_input.iloc[test_idx,-50:].reset_index(drop = True)
    
    return X_train, X_valid, X_test,  y_train, y_valid, y_test, X_train_word2vec, X_valid_word2vec, X_test_word2vec




def get_data_splits_old_algo_doc_word(data_input, features_input, idx):
    
    X_train, X_test, y_train, y_test = train_test_split(features_input, data_input["high_level_substr"], test_size = 0.3,
                                                     shuffle = True, stratify = data_input["high_level_substr"].values)
    
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.33,
                                                     shuffle = True, stratify = y_test.values)
    
    train_idx, valid_idx, test_idx = y_train.index, y_valid.index, y_test.index
    
    
    y_train, y_valid, y_test = (y_train.values == "pectin").astype(float), (y_valid.values == "pectin").astype(float), (y_test.values == "pectin").astype(float)
    
    X_train_word2vec, X_valid_word2vec, X_test_word2vec  = data_input.iloc[train_idx,-100:].reset_index(drop = True), data_input.iloc[valid_idx,-100:].reset_index(drop = True), data_input.iloc[test_idx,-100:].reset_index(drop = True)
    
    return X_train, X_valid, X_test,  y_train, y_valid, y_test, X_train_word2vec, X_valid_word2vec, X_test_word2vec



def get_data_splits_old_algo_real(data_input, features_input, idx):
    
    X_train, X_test, y_train, y_test = train_test_split(features_input, data_input["high_level_substr"], test_size = 0.3,
                                                     shuffle = True, stratify = data_input["high_level_substr"].values)
    
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.33,
                                                     shuffle = True, stratify = y_test.values)
    
    train_idx, valid_idx, test_idx = y_train.index, y_valid.index, y_test.index
    
    
    y_train, y_valid, y_test = (y_train.values == "pectin").astype(float), (y_valid.values == "pectin").astype(float), (y_test.values == "pectin").astype(float)
    
    X_train_word2vec, X_valid_word2vec, X_test_word2vec  = data_input.iloc[train_idx,-100:].reset_index(drop = True), data_input.iloc[valid_idx,-100:].reset_index(drop = True), data_input.iloc[test_idx,-100:].reset_index(drop = True)
    
    return X_train, X_valid, X_test,  y_train, y_valid, y_test, X_train_word2vec, X_valid_word2vec, X_test_word2vec







def get_data_splits_new_algo(data_input, features_input, idx):
    
    X_train, X_test, y_train, y_test = train_test_split(features_input, data_input["high_level_substr"], test_size = 0.3,
                                                     shuffle = True, stratify = data_input["high_level_substr"].values)
    
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.33,
                                                     shuffle = True, stratify = y_test.values)
    
    train_idx, valid_idx, test_idx = y_train.index, y_valid.index, y_test.index
    
    
    y_train, y_valid, y_test = (y_train.values == "pectin").astype(float), (y_valid.values == "pectin").astype(float), (y_test.values == "pectin").astype(float)
    
    X_train_word2vec, X_valid_word2vec, X_test_word2vec  = data_input.iloc[train_idx,-50:].reset_index(drop = True), data_input.iloc[valid_idx,-50:].reset_index(drop = True), data_input.iloc[test_idx,-50:].reset_index(drop = True)
    
    return X_train, X_valid, X_test,  y_train, y_valid, y_test, X_train_word2vec, X_valid_word2vec, X_test_word2vec






def first_LSTM_training(catch, model_cbow, idx):

    model_word2vec = attention_lstm_model(False, model_cbow)
    
    model_word2vec = simple_lstm(False, model_cbow)
    
    model_word2vec.fit(catch[idx][0], catch[idx][3], epochs = 2000, verbose = 0, 
                  callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 30,
                                                              restore_best_weights=True), 
                  validation_data = (catch[idx][1], catch[idx][4]))
    
    X_train_logits = model_word2vec.predict(catch[idx][0], verbose = 0)
    X_valid_logits = model_word2vec.predict(catch[idx][1], verbose = 0)
    X_test_logits = model_word2vec.predict(catch[idx][2], verbose = 0)
    
    train_acc = model_word2vec.evaluate(catch[idx][0], catch[idx][3], verbose = 0)[1]
    valid_acc = model_word2vec.evaluate(catch[idx][1], catch[idx][4], verbose = 0)[1]
    test_acc = model_word2vec.evaluate(catch[idx][2], catch[idx][5], verbose = 0)[1]
    
    return X_train_logits, X_valid_logits, X_test_logits, train_acc, valid_acc, test_acc


def get_targets_with_weights(batch_data, initial_ensembles, log_sigma_points_1): 

    n_hidden_1 = len(weights_ann_1[0].ravel())

    hidden_weights_1 = initial_ensembles[:,:n_hidden_1].reshape( size_ens, batch_data.shape[1], h1)

    
    hidden_output_1 = np.einsum('ij,kjl->kil', batch_data, hidden_weights_1)

    
    hidden_layer_bias_1 = initial_ensembles[:,n_hidden_1:(n_hidden_1 + h1)].reshape(size_ens, 1,  h1)



    hidden_output_1 = hidden_output_1 + hidden_layer_bias_1

    n_pred_weights_1 = len(weights_ann_1[2].ravel())

    output_weights_1 = initial_ensembles[:,(n_hidden_1 + h1):(n_hidden_1 + h1 + n_pred_weights_1) ].reshape(size_ens, h1, 1)

    output_1 = np.einsum('ijk,ikl->ijl', hidden_output_1, output_weights_1)


    output_layer_bias_1 = initial_ensembles[:,(n_hidden_1 + h1 + n_pred_weights_1):(n_hidden_1 + h1 + n_pred_weights_1 + 1)].reshape(size_ens, 1, 1)



    final_output_1 = output_1 + output_layer_bias_1

    n_hidden_2 = len(weights_ann_2[0].ravel())

    initial_ensembles_1 = initial_ensembles.copy()[:, total_weights_1:(total_weights_1+ total_weights_2)]

    hidden_weights_2 = initial_ensembles_1[:,:n_hidden_2].reshape(size_ens, batch_data.shape[1], h2)



    hidden_output_2 = np.einsum('ij,kjl->kil', batch_data, hidden_weights_2)

    hidden_layer_bias_2 = initial_ensembles[:,n_hidden_2:(n_hidden_2 + h2)].reshape(size_ens, 1,  h2)

    hidden_output_2 = hidden_output_2+ hidden_layer_bias_2

    n_pred_weights_2 = len(weights_ann_2[2].ravel())

    output_weights_2 = initial_ensembles_1[:,(n_hidden_2 + h2):(n_hidden_2 + h2 + n_pred_weights_2) ].reshape(size_ens, h2, 1)


    output_2 = np.einsum('ijk,ikl->ijl', hidden_output_2, output_weights_2)


    output_layer_bias_2 = initial_ensembles_1[:,(n_hidden_2 + h2 + n_pred_weights_2):(n_hidden_2 + h2 + n_pred_weights_2 + 1)].reshape(size_ens, 1, 1)


    final_output_2 = output_2 + output_layer_bias_2


    weights_1 = initial_ensembles[:, :total_weights_1]

    weights_2 = initial_ensembles[:, total_weights_1:(total_weights_1 + total_weights_2)]


    avg_weights = initial_ensembles[:, -1].reshape(-1,1)

    avg_weights_sig = expit(avg_weights)
    
    avg_weights_sig = avg_weights_sig.reshape(avg_weights_sig.shape[0], 1, avg_weights_sig.shape[1])
    
    complement_weights_sig = 1 - expit(avg_weights)
    
    complement_weights_sig = complement_weights_sig.reshape(complement_weights_sig.shape[0], 1, complement_weights_sig.shape[1])

    final_output_1 = final_output_1*complement_weights_sig
    
    final_output_2 = final_output_2*avg_weights_sig
    
    output_1_ravel = final_output_1.reshape(size_ens, final_output_1.shape[1]*final_output_1.shape[2])

    output_2_ravel = final_output_2.reshape(size_ens, final_output_2.shape[1]*final_output_2.shape[2])


    output_1_ravel = output_1_ravel

    output_2_ravel = output_2_ravel



    weights_1_add = np.zeros((size_ens, (total_weights_2 - total_weights_1)))



    weights_1 = np.hstack((weights_1, weights_1_add))
    


    stack_1 = np.hstack((output_1_ravel, weights_1, np.repeat(0, size_ens).reshape(-1,1), np.repeat(0, size_ens).reshape(-1,1)))



    
    stack_2 = np.hstack((output_2_ravel, weights_2, avg_weights, log_sigma_points_1))

    
    initial_aug_state = np.hstack((stack_1, stack_2)) 
    

    return initial_aug_state , output_1_ravel, output_2_ravel, log_sigma_points_1


def rep_one(idx, X_train_logits, X_valid_logits, X_test_logits, inflation_factor = 2): 
    catch_1 = []
    catch_2 = []
    catch_3 = []
    catch_4 = []
    catch_5 = []
#     from scipy.special import expit
    patience_smaller = 0
# patience_bigger = 0

    best_train_acc = 0
    best_valid_acc = 0

    best_valid_mae = 10

    ## create training batch chunks
    train_idx = list(range(0, X_train.shape[0]))
    batch_chunks = [train_idx[i:i+batch_size] for i in range(0,len(train_idx),batch_size)]

    ## generate some augmented variable for iteration 0
    initial_aug_state_mean = np.repeat(0, total_weights)
    initial_aug_state_mean = initial_aug_state_mean.reshape(-1,1)

    initial_aug_state_cov = var_weights*np.identity((total_weights))
    initial_ensembles = mvn(initial_aug_state_mean.reshape(initial_aug_state_mean.shape[0],), initial_aug_state_cov).rvs(size = size_ens)

#     initial_ensembles[:,-1] = initial_betas
    
    log_sigma_points_1 = (np.log(gamma(100, scale = 1/100).rvs(size_ens))).reshape(size_ens, 1)
    
#     log_sigma_points_2 = np.repeat(0, size_ens).reshape(size_ens,1)
    
    logit_measurement_noise_train = mvn(np.repeat(0,X_train_logits.shape[1]), var_targets*np.identity(X_train_logits.shape[1])).rvs(X_train_logits.shape[0])

    logit_measurement_noise_valid = mvn(np.repeat(0,X_valid_logits.shape[1]), var_targets*np.identity(X_valid_logits.shape[1])).rvs(X_valid_logits.shape[0])

    logit_measurement_noise_test = mvn(np.repeat(0,X_test_logits.shape[1]), var_targets*np.identity(X_test_logits.shape[1])).rvs(X_test_logits.shape[0])

    X_train_logits = X_train_logits + logit_measurement_noise_train.reshape(-1,1)

    X_valid_logits = X_valid_logits + logit_measurement_noise_valid.reshape(-1,1)

    X_test_logits = X_test_logits + logit_measurement_noise_test.reshape(-1,1)

    X_train_probs = expit(X_train_logits)

    X_valid_probs = expit(X_valid_logits)

    X_test_probs = expit(X_test_logits)
    
    print(X_train_probs.shape)

    y_train = (X_train_probs >= 0.5).astype(float)

    y_valid = (X_valid_probs >= 0.5).astype(float)

    y_test = (X_test_probs >= 0.5).astype(float)

    for iter1 in range(0,500):

        for batch_idx in batch_chunks:

            batch_data = X_train_word2vec.iloc[batch_idx,:]
            batch_targets = X_train_logits[batch_idx,:]
            batch_targets = batch_targets.ravel().reshape(-1,1)

            column_mod_2_shape = total_weights_2 + batch_data.shape[0]*1 + 1 + 1
        
            H_t = np.hstack((np.identity(batch_targets.shape[0]), np.zeros((batch_targets.shape[0], column_mod_2_shape-batch_targets.shape[0]))))

            current_aug_state, column_mod_1, column_mod_2, log_sigma_points_1 = get_targets_with_weights(batch_data, initial_ensembles, log_sigma_points_1)
            
            var_targets_vec = np.log(1 + np.exp(log_sigma_points_1))
            
            var_targets_vec = var_targets_vec
            
            current_aug_state_var = np.cov(current_aug_state.T) + inflation_factor*np.identity(current_aug_state.shape[1])

            
            G_t = np.array([1 , 1]).reshape(-1,1)
            
            scirpt_H_t = np.kron(G_t.T, H_t)
        
            for ensemble_idx in range(0, current_aug_state.shape[0]):
                
                var_targets1 = var_targets_vec[ensemble_idx,:]
                

                
                R_t = var_targets1*np.identity(batch_targets.shape[0])
            
                measurement_error = mvn(np.repeat(0,batch_targets.shape[0]), var_targets1*np.identity(batch_targets.shape[0])).rvs(1).reshape(-1,1)
            
                target_current = batch_targets + measurement_error
                
                K_t = current_aug_state_var@scirpt_H_t.T@np.linalg.inv(scirpt_H_t@current_aug_state_var@scirpt_H_t.T + R_t)

                current_aug_state[ensemble_idx,:] = current_aug_state[ensemble_idx,:] +(K_t@(target_current -scirpt_H_t@current_aug_state[ensemble_idx,:].reshape(-1,1))).reshape(current_aug_state.shape[1],)
        

            weights_ann_1 = current_aug_state[:,batch_targets.shape[0]:(batch_targets.shape[0] + total_weights_1)]      

            weights_ann_2 = current_aug_state[:,-(total_weights_2+1):-2]    

            initial_ensembles = np.hstack((weights_ann_1, weights_ann_2, current_aug_state[:,-2].reshape(-1,1)))
            
            log_sigma_points_1 = current_aug_state[:,-1].reshape(-1,1)
               
            avg_betas = expit(current_aug_state[:,-2])
        
            complement = 1-avg_betas

            
            current_aug_state1, column_mod_11, column_mod_21, log_sigma_points_1 = get_targets_with_weights(X_train_word2vec, initial_ensembles, log_sigma_points_1)
            
            initial_targets = column_mod_11 + column_mod_21
            
            
            initial_targets = initial_targets.reshape(size_ens, X_train_word2vec.shape[0],1)
            
            initial_targets_train = initial_targets
            
            
#             ind = (X_train_logits_true >= np.percentile(initial_targets_train, axis = 0, q = (2.5, 97.5))[0,:,:]) & (X_train_logits_true <= np.percentile(initial_targets_train, axis = 0, q = (2.5, 97.5))[1,:,:])
        
            initial_targets_softmax = expit(initial_targets)
            
            ind = (X_train_probs_true >= np.percentile(initial_targets_softmax, axis = 0, q = (2.5, 97.5))[0,:,:]) & (X_train_probs_true <= np.percentile(initial_targets_softmax, axis = 0, q = (2.5, 97.5))[1,:,:])
            
            
            initial_targets_softmax_mean = np.mean(initial_targets_train,0)
            
            initial_targets_softmax_std = np.std(initial_targets_train,0)
             
            coverage = np.mean(ind.ravel())
        
            initial_targets = np.mean(initial_targets,0)
            
        
            train_mae_logits = np.mean(np.abs(X_train_logits.ravel() - initial_targets.ravel()))
        
            initial_targets = expit(initial_targets)
        
            train_mae = np.mean(np.abs(X_train_probs.ravel() - initial_targets.ravel()))
        

            pred_ohe = (initial_targets >= 0.5).astype(float)
            y_train_curr = y_train
            acc = np.mean(pred_ohe == y_train_curr)
            
            predicted_batch_1 = column_mod_11
            predicted_batch_2 = column_mod_21
            
            predicted_batch_1_ind = predicted_batch_1.reshape(size_ens, X_train_word2vec.shape[0], 1)
            
            predicted_batch_1_ind_train = predicted_batch_1_ind

            predicted_batch_1_ind = np.mean(predicted_batch_1_ind,0)
            predicted_batch_1_ind = expit(predicted_batch_1_ind)
        
        
            predicted_batch_2_ind = predicted_batch_2.reshape(size_ens, X_train_word2vec.shape[0], 1)
            predicted_batch_2_ind_train = predicted_batch_2_ind

            predicted_batch_2_ind = np.mean(predicted_batch_2_ind,0)
            predicted_batch_2_ind = expit(predicted_batch_2_ind)
        
            predicted_batch_1_ind = (predicted_batch_1_ind >= 0.5).astype(float)
            predicted_batch_2_ind = (predicted_batch_2_ind >= 0.5).astype(float)
       
            acc_ind_1_train = np.mean(predicted_batch_1_ind == y_train_curr)
            acc_ind_2_train = np.mean(predicted_batch_2_ind == y_train_curr)
        
            acc_ind_1_train_idx =  (predicted_batch_1_ind == y_train_curr).nonzero()
            acc_ind_2_train_idx =  (predicted_batch_2_ind == y_train_curr).nonzero()
        
            common_correct = len(set(acc_ind_1_train_idx[0]).intersection(acc_ind_2_train_idx[0]))/len(predicted_batch_1_ind)
        
            current_aug_state1, column_mod_11, column_mod_21, log_sigma_points_1 = get_targets_with_weights(X_valid_word2vec, initial_ensembles, log_sigma_points_1)
            
            initial_targets = column_mod_11 + column_mod_21
            
            initial_targets = initial_targets.reshape(size_ens, X_valid_word2vec.shape[0],1)

        
            initial_targets = np.mean(initial_targets,0)
        
            valid_mae_logits = np.mean(np.abs(X_valid_logits.ravel() - initial_targets.ravel()))
        
            initial_targets = expit(initial_targets)
        
            valid_mae = np.mean(np.abs(X_valid_probs.ravel() - initial_targets.ravel()))
        
            pred_ohe = (initial_targets >= 0.5).astype(float)
            y_train_curr = y_valid
            acc_valid = np.mean(pred_ohe == y_train_curr)
            
            predicted_batch_1 = column_mod_11
            predicted_batch_2 = column_mod_21
            
            predicted_batch_1_ind = predicted_batch_1.reshape(size_ens, X_valid_word2vec.shape[0], 1)
            predicted_batch_1_ind = np.mean(predicted_batch_1_ind,0)
            predicted_batch_1_ind = expit(predicted_batch_1_ind)
        
            predicted_batch_2_ind = predicted_batch_2.reshape(size_ens, X_valid_word2vec.shape[0], 1)
            predicted_batch_2_ind = np.mean(predicted_batch_2_ind,0)
            predicted_batch_2_ind = expit(predicted_batch_2_ind)
        
            predicted_batch_1_ind = (predicted_batch_1_ind >= 0.5).astype(float)
            predicted_batch_2_ind = (predicted_batch_2_ind >= 0.5).astype(float)
         
            acc_ind_1_valid = np.mean(predicted_batch_1_ind == y_train_curr)
            acc_ind_2_valid = np.mean(predicted_batch_2_ind == y_train_curr)
        
            current_aug_state1, column_mod_11, column_mod_21, log_sigma_points_1 = get_targets_with_weights(X_test_word2vec, initial_ensembles, log_sigma_points_1)
            
            initial_targets = column_mod_11 + column_mod_21
            
            initial_targets = initial_targets.reshape(size_ens, X_test_word2vec.shape[0],1)
            
            initial_targets_test = initial_targets
   
            initial_targets = np.mean(initial_targets,0)

            test_mae_logits = np.mean(np.abs(X_test_logits.ravel() - initial_targets.ravel()))
        
            initial_targets = expit(initial_targets)
        
            test_mae = np.mean(np.abs(X_test_probs.ravel() - initial_targets.ravel()))
        
            pred_ohe = (initial_targets >= 0.5).astype(float)

            y_train_curr = y_test
            acc_test = np.mean(pred_ohe == y_train_curr)        

            enkf_norm = np.linalg.norm(initial_ensembles.mean(0))   
            
            catch_1.append(var_targets_vec.mean().round(4))
            catch_2.append(var_targets_vec.std().round(4))
            catch_3.append(np.mean(initial_targets_softmax_std.ravel()**2))
            catch_4.append(np.std(initial_targets_softmax_std.ravel()**2))
            catch_5.append(coverage)
                            
        if acc_valid > best_valid_acc:
            best_train_acc = acc
            best_valid_acc = acc_valid
            best_test_acc = acc_test

            best_complement = np.mean(complement)
            best_avg_betas = np.mean(avg_betas)

            best_train_mae = train_mae
            best_valid_mae = valid_mae
            best_test_mae = test_mae
            best_coverage = coverage
#             best_initial_targets = initial_targets
#             best_X_train_probs_vec = X_train_probs_vec
            best_initial_targets_train = initial_targets_train
            best_predicted_batch_1_ind_train = predicted_batch_1_ind_train
            best_predicted_batch_2_ind_train = predicted_batch_2_ind_train
            best_var_targets = var_targets_vec
            best_initial_targets_test = initial_targets_test
            patience_smaller = 0 
                
        elif acc_valid <= best_valid_acc: 
            patience_smaller += 1
            
        else:
            pass
        
        
        
        print("epoch "+ str(iter1))
        print("patience "+ str(patience_smaller))
        print("coverage is "+ str(coverage))
        print("train accuracy "+ str(acc.round(3)))
        print("valid accuracy "+ str(acc_valid.round(3)))
        print("test accuracy "+ str(acc_test.round(3))) 
        print("individual train accuracy "+ str(acc_ind_1_train.round(3)) + " & " + str(acc_ind_2_train.round(3)))
        print("individual weights "+ str(np.mean(complement).round(3)) + " & " + str(np.mean(avg_betas).round(3)))
        print("individual valid accuracy "+ str(acc_ind_1_valid.round(3)) + " & " + str(acc_ind_2_valid.round(3)))
        print("train emulator mae " + str(train_mae))
        print("valid emulator mae " + str(valid_mae))
        print("test emulator mae " + str(test_mae))
        print("train emulator logits mae " + str(train_mae_logits))
        print("valid emulator logits mae " + str(valid_mae_logits))
        print("test emulator logits mae " + str(test_mae_logits))    
        print("var train logits preds"+  str(np.mean(initial_targets_softmax_std.ravel()**2)))
        print("std norm of ensembles " + str(np.linalg.norm(initial_ensembles, axis = 1).std()))
        print(var_targets_vec.mean().round(4), var_targets_vec.std().round(4))
#         print(var_targets_vec.mean().round(4), var_targets_vec.std().round(4))
        
      
    
        if (patience_smaller > threshold):
            break 
        else:
            pass

    return [initial_targets_train, initial_targets_test ,var_targets_vec, catch_1, catch_2, catch_3, catch_4, catch_5]

def ann(hidden , vec_size): 
    input_layer = tf.keras.layers.Input(shape = (vec_size))
    hidden_layer = tf.keras.layers.Dense(hidden)
    hidden_output = hidden_layer(input_layer)
    pred_layer = tf.keras.layers.Dense(1)
    pred_output = pred_layer(hidden_output)
#     pred_output = tf.keras.layers.Activation("softmax")(pred_output)
    model = tf.keras.models.Model(input_layer, pred_output)
    return model


def non_recurrent_attention_model(training): 
    padding_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    unknown_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    weight_vectors = np.vstack((padding_vector, unknown_vector))
    weight_vectors = np.vstack((weight_vectors, model_cbow.wv.vectors))
    embedding_layer = tf.keras.layers.Embedding(len(weight_vectors),
                            weight_vectors.shape[1],
                            weights=[weight_vectors],
                            mask_zero = False,
                            trainable=False)
    
    vectorize_layer = tf.keras.layers.TextVectorization(
                     output_mode='int',
                     vocabulary=model_cbow.wv.index_to_key, 
                     standardize = None)
    input_layer = tf.keras.layers.Input(shape = (1,), dtype = tf.string)

    vectorize = vectorize_layer(input_layer)

    vectorize.shape

    emb_output = embedding_layer(vectorize)

    emb_output.shape

    x_a = tf.keras.layers.Dense(emb_output.get_shape()[-1]//2, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp")(emb_output) 
    
    x_a = tf.keras.layers.Dropout(0.5)(x_a, training = training)
    
    x_a = tf.keras.layers.Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context")(x_a)

    x_a = tf.keras.layers.Flatten()(x_a)

    att_out = tf.keras.layers.Activation('softmax')(x_a) 

    x_a2 = tf.keras.layers.RepeatVector(emb_output.get_shape()[-1])(att_out)

    x_a2 = tf.keras.layers.Permute([2,1])(x_a2)

    out = tf.keras.layers.Multiply()([emb_output,x_a2])
    
    out = tf.keras.layers.Lambda(lambda x : tf.math.reduce_sum(x, axis = 1), name='expectation_over_words')(out)

    dropout_layer = tf.keras.layers.Dropout(0.65)(out, training = training)

    pred_head = tf.keras.layers.Dense(num_classes)

    pred_output = pred_head(dropout_layer)

    model = tf.keras.models.Model(input_layer, pred_output)
    
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                 metrics= "accuracy")
    
    return model
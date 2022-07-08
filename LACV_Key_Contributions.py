    ##############################
    ##########   LAFA   ##########
    ##############################

    def building_LAFA_module(self, xyzrgb, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyzrgb = self.loc_info_encoding_unit(xyz, neigh_idx)
        
        f_xyzrgb = helper_tf_util.conv2d(f_xyzrgb, d_in, [1, 1], name + "mlp1", [1, 1], "VALID", True, is_training)
        f_neighbours = self.loc_info_encoding_unit(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyzrgb], axis=-1)
        f_loc_adap, _ = self.adap_aug_unit(f_concat, d_out // 2, name + "adap_aug_1", is_training)

        f_xyzrgb = helper_tf_util.conv2d(f_xyzrgb, d_out // 2, [1, 1], name + "mlp2", [1, 1], "VALID", True, is_training)
        f_neighbours = self.loc_info_encoding_unit(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyzrgb], axis=-1)
        f_loc_adap, f_weighted = self.adap_aug_unit(f_concat, d_out, name + "adap_aug_2", is_training)
        
        neigh = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        neigh = neigh + f_weighted
        return f_loc_adap, neigh

    def loc_info_encoding_unit(self, x, neigh_idx):
        tile_x = tf.tile(tf.expand_dims(x, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])  # B, N, k, d_in
        neigh_x = self.gather_neighbour(x, neigh_idx)  # B, N, k, d_in
        x_info = tile_f - neigh_f
        return x_info  # B, N, k, d_in

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def adap_aug_unit(feature, d_out, name, is_training):
        k_weights = helper_tf_util.conv2d(feature, feature.get_shape()[-1].value, [1, 1], name + 'weight', [1, 1], 'VALID', bn=False, activation_fn=None)
        k_weights = tf.nn.softmax(k_weights, axis=2)
        f_max = tf.reduce_max(feature, axis=-2, keepdims=True)
        f_weighted = feature * k_weights
        f_adap = tf.reduce_sum(f_weighted, axis=-2, keepdims=True)
        f_agg = tf.concat([f_max, f_adap], axis=-1)
        f_agg = helper_tf_util.conv2d(f_adap, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg, f_weighted



    ##############################
    #########   C-VLAD   #########
    ##############################

    for k in range(self.config.num_layers):
        n = num_list[-k - 1]
        vlad = self.G_VLAD(f_encoder_list[-k - 1], f_encoder_list[-1].get_shape()[3].value // 4, n, 'Encoder_layer_G_' + str(k), is_training)
        vlad = tf.tile(tf.expand_dims(vlad, axis=1), [1, num_list[-1], 1])
        vlad = tf.expand_dims(vlad, axis=2)
        if k == 0:
            feature = vlad
        else:
            feature = tf.concat([feature, vlad], axis=-1)
    
    feature = tf.concat([feature, f_encoder_list[-1]], axis=-1)
    feature = helper_tf_util.conv2d(feature, f_encoder_list[-1].get_shape()[3].value, [1, 1], 'decoder_0', [1, 1], 'VALID', True, is_training)

    
    def VLAD(self, feature, d_out, num_points, name, is_training):
        n_size = num_points
        f_size = feature.get_shape()[-1].value
        clus_num = d_out // 16
        feature = tf.reshape(feature, [-1, f_size])
        feature = tf.nn.l2_normalize(feature, -1)

        cluster_weights = tf.get_variable(
            name + "cluster_weights",
            [f_size, clus_num],
            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(f_size)))
        activation = tf.matmul(feature, cluster_weights)

        cluster_biases = tf.get_variable(
            name + "cluster_biases",
            [clus_num],
            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(f_size)))
        activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        activation = tf.reshape(activation, [-1, n_size, clus_num])
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable(
            name + "cluster_weights2",
            [1, f_size, clus_num],
            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(f_size)))
        a = tf.multiply(a_sum, cluster_weights2)
        activation = tf.transpose(activation, perm=[0, 2, 1])
        reshaped_input = tf.reshape(feature, [-1, n_size, f_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, clus_num * f_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        hidden1_weights = tf.get_variable(
            name + "hidden1_weights",
            [clus_num * f_size, d_out],
            initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(clus_num)))
        vlad = tf.matmul(vlad, hidden1_weights)
        return vlad



    ##############################
    #####  aggre loss func   #####
    ##############################

    with tf.variable_scope('loss'):           
        aug_loss_weights = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2])
        aug_loss = 0
        for i in range(self.config.num_layers):
            centroids = tf.reduce_mean(self.new_feat_list[i], axis=2)
            # print(centroids.shape, self.feat_list[i].shape)
            relative_dis = tf.reduce_sum(tf.nn.softmax(centroids, axis=-1) - tf.nn.softmax(self.feat_list[i], axis=-1), axis=-1)
            aug_loss = aug_loss + aug_loss_weights[i] * tf.reduce_mean(tf.reduce_mean(relative_dis, axis=-1), axis=-1)

        self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights) + aug_loss

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss
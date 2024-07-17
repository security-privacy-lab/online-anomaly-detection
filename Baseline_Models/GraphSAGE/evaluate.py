import tensorflow as tf
import argparse
from graphsage.supervised_train import construct_placeholders, SupervisedGraphsage, UniformNeighborSampler, SAGEInfo, NodeMinibatchIterator, load_data, incremental_evaluate
import numpy as np

# Define argument parser
parser = argparse.ArgumentParser(description='Evaluate GraphSAGE model on new data.')
parser.add_argument('--train_prefix', required=True, help='Prefix identifying training data')
parser.add_argument('--model', default='graphsage_mean', help='Model name')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
parser.add_argument('--max_degree', type=int, default=128, help='Maximum node degree')
parser.add_argument('--samples_1', type=int, default=25, help='Number of samples in layer 1')
parser.add_argument('--samples_2', type=int, default=10, help='Number of samples in layer 2')
parser.add_argument('--samples_3', type=int, default=0, help='Number of samples in layer 3')
parser.add_argument('--dim_1', type=int, default=128, help='Size of output dim (final is 2x this, if using concat)')
parser.add_argument('--dim_2', type=int, default=128, help='Size of output dim (final is 2x this, if using concat)')
parser.add_argument('--identity_dim', type=int, default=0, help='Identity embedding dimension (if positive)')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--model_size', default='small', help='Model size')
parser.add_argument('--sigmoid', action='store_true', help='Use sigmoid loss')
parser.add_argument('--log_device_placement', action='store_true', help='Log device placement')

args = parser.parse_args()

# Function to evaluate new data
def evaluate_new_data(new_data_prefix):
    new_data = load_data(new_data_prefix)

    G = new_data[0]
    features = new_data[1]
    id_map = new_data[2]
    class_map = new_data[4]
    num_classes = len(set(class_map.values()))

    if features is not None:
        features = np.vstack([features, np.zeros((features.shape[1],))])

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=args.batch_size,
            max_degree=args.max_degree,
            context_pairs=None)

    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    # Load model
    if args.model == 'graphsage_mean':
        sampler = UniformNeighborSampler(adj_info)
        if args.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1),
                                SAGEInfo("node", sampler, args.samples_2, args.dim_2),
                                SAGEInfo("node", sampler, args.samples_3, args.dim_2)]
        elif args.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1),
                                SAGEInfo("node", sampler, args.samples_2, args.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos, 
                                     model_size=args.model_size,
                                     sigmoid_loss=args.sigmoid,
                                     identity_dim=args.identity_dim,
                                     logging=True)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=args.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, "./model/model.ckpt")
    print("Model restored.")

    # Evaluate
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    sess.run(val_adj_info.op)
    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, args.batch_size)
    print("Validation stats:",
                  "loss=", "{:.5f}".format(val_cost),
                  "f1_micro=", "{:.5f}".format(val_f1_mic),
                  "f1_macro=", "{:.5f}".format(val_f1_mac),
                  "time=", "{:.5f}".format(duration))

    sess.close()

if __name__ == "__main__":
    evaluate_new_data(args.train_prefix)

import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from target_lstm import TARGET_LSTM
import pickle
import os
## Ignore TensorFlow logging
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 64

PRE_GEN_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs 120 -- > 1
PRE_DIS_EPOCH_NUM = 5   # 50 -> 1
IN_DIS_EPOCH = 1 # 3 -> 1
ADV_GEN_EPOCH_NUM = 3
ADV_DIS_EPOCH_NUM = 1 # 5 -> 1
VOCAB_SIZE = 5000
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 100
positive_file = 'tmp/real_data.txt'
negative_file = 'tmp/generator_sample.txt'
## 用来保存分割句子的文件
positive_file_split = 'tmp/real_data.split.txt'
negative_file_split = 'tmp/generator_sample.split.txt'
eval_file = 'tmp/eval_file.txt'
generated_num = 10000
save_path = 'experiments/model4/'
LOG_FILE = os.path.join(save_path, 'experiment-log.txt')
TARGET_PARAMS = 'save/target_params_py3.pkl'
restore_from = None


if not os.path.exists(save_path):
    os.mkdir(save_path)
else:
    restore_from = tf.train.latest_checkpoint(save_path)

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


## 将文件中的句子分割成不同长度句子
def split_sentence_file(input_file, output_file):
    # Load data
    print("start split for file :", input_file)
    datasets = []
    with open(input_file)as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            for i in range(1, SEQ_LENGTH+1):
                data = parse_line[:i] + [0] * (SEQ_LENGTH-i)
                datasets.append(data)
    with open(output_file, 'w') as fout:
        for data in datasets:
            buffer = ' '.join([str(x) for x in data]) + '\n'
            fout.write(buffer)
    print("success!!!!!!")

## 将句子分割成不同长度片段 !注意, 这里应该将句子分成 SEQ_LENGTH个, 一个单词也有奖励
# , 一个单词分类器可以判断这个单词在开头是不是合理.
def split_sentence(input_data):
    """
    input_data: numpy.array with [batch_size x seq_length]
    """
    # make sure this is 2d array
    assert input_data.ndim == 2
    # Load data 
    # TO-DO better padding with numpy
    datasets = []
    for line in input_data:
        for i in range(1, SEQ_LENGTH+1):
            data = np.pad(line[:i], (0, SEQ_LENGTH-i), 'constant')
            datasets.append(data)
    return datasets
  
# 将句子分段, 并从判别器获取奖励
def get_rewords_from_discriminator(sess, input_x, discriminator):
    """computing rewards for a batch of sentenses
    Input:
        sess: a TensorFlow Session
        input_x: [batch_size x seq_length], a batch of generated sentences
        discriminator: a discriminator object
    Return:
        rewards: the rewards of input_x, [batch_size x seq_length]
    """
    rewards = []  # batch_size x seq_length
    split_data = split_sentence(input_x) # split data as [SEQ_LENGTH*BATCH_SIZE, SEQ_LENGTH]
    for given_num in range(1, SEQ_LENGTH+1):
        batch_data = [] # batch_size x seq_length
        for i in range(BATCH_SIZE):
            batch_data.append(split_data[i*SEQ_LENGTH+given_num-1])
        feed = {discriminator.input_x: batch_data, discriminator.dropout_keep_prob: 1.0}
        ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
        ypred = np.array([item[1] for item in ypred_for_auc])
        rewards.append(ypred) # seq_length x batch_size
    rewards = np.transpose(np.array(rewards)) # batch_size x seq_length
    return rewards

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = pickle.load(open(TARGET_PARAMS, 'rb'))
    target_lstm = TARGET_LSTM(VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=VOCAB_SIZE, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)


    saver = tf.train.Saver() # saver
    # 开始 Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open(LOG_FILE, 'w')
    #  pre-train generator
    print ('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_GEN_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print ('pre-train epoch ', epoch, 'test_loss ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)

    print ('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    ## 将真实数据与生成数据都切分成不同段的数据来训练判别器
    split_sentence_file(positive_file, positive_file_split)
    for epoch in range(PRE_DIS_EPOCH_NUM): 
        print("EPOCH : %d  $$$$$$$$$$$" % epoch)
        print("Generating and Spliting Negative file.......")
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        split_sentence_file(negative_file, negative_file_split)
        print("Load file to loader.....")
        dis_data_loader.load_train_data(positive_file_split, negative_file_split)
        print("Start training ...... ")
        for ep in range(IN_DIS_EPOCH): # 3 --> 1
            print("inner epoch: %d :" % ep)
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                ## 获取判别器loss, 以观察进度
                loss, _ = sess.run([discriminator.loss, discriminator.train_op], feed)
                if it % 1000 == 0:
                    print (f'Total Epoch {epoch}, Gen Epoch {ep}, steps {it}, loss {loss}')

    print ('#########################################################################')
    print ('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        print(f"Total batch {total_batch} ------------------------------------------")
        # Train the generator for one step
        for it in range(ADV_GEN_EPOCH_NUM):
            samples = generator.generate(sess)
            # 修改reward获取方式, 改为从判别器直接获取各个段的rewards
            rewards = get_rewords_from_discriminator(sess, samples, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print ('total_batch: ', total_batch, 'test_loss: ', test_loss)
            log.write(buffer)

        # Train the discriminator
        for epoch in range(ADV_DIS_EPOCH_NUM): # 5 --> 1
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            split_sentence_file(negative_file, negative_file_split)
            dis_data_loader.load_train_data(positive_file_split, negative_file_split)
            for ep in range(IN_DIS_EPOCH): # 3 --> 1
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    ## 获取判别器loss, 以观察进度
                    loss, _ = sess.run([discriminator.loss, discriminator.train_op], feed)
                    if it % 1000 == 0:
                        print (f'Total Epoch {epoch}, Gen Epoch {ep}, steps {it}, loss {loss}')
        # Save model 
        path = os.path.join(save_path, 'after-epoch')
        saver.save(sess, path, global_step=total_batch+1)
    log.close()


if __name__ == '__main__':
    main()

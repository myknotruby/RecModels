import numpy as np
import sys
import tensorflow.compat.v1 as tf
import argparse
import sys
import os
import heapq
import math

tf.disable_eager_execution()
#comment
class Dataset():
    def __init__(self, filename):
        self.data, self.shape = self.getData(filename)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()

    def getData(self, filename):
        if filename == 'ml-1m':
            print('my baby, now loading ml-1m data')
            data = []
            filepath = '/sdb1/datasets/ml-1m/ratings.dat'
            num_u = 0  # 用户个数
            num_i = 0  # 电影个数
            max_rating = 0.0  # 最大评分
            with open(filepath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split("::")
                        user_id = int(lines[0])
                        movie_id = int(lines[1])
                        score = float(lines[2])
                        time = int(lines[3])
                        data.append((user_id, movie_id, score, time))
                        if user_id > num_u:
                            num_u = user_id
                        if movie_id > num_i:
                            num_i = movie_id
                        if score > max_rating:
                            max_rating = score
            self.maxRate = max_rating
            print("users number : {}----- movies number : {}----- max Rating : "
                  "{}----- Data size : {} ".format(num_u, num_i, self.maxRate, len(data)))
            return data, [num_u, num_i]
        else:
            print("Don't find the file")
            sys.exit()

    def getTrainTest(self):
        data = self.data
        data = sorted(data, key=lambda x: (x[0], x[3]))  # 按照用户数、评论时间排序
        train = []
        test = []
        for i in range(len(data) - 1):  # 每个用户的最后一条数据作为评论数据
            user = data[i][0] - 1
            movie = data[i][1] - 1
            rate = data[i][2]
            if data[i][0] != data[i + 1][0]:
                test.append((user, movie, rate))
            else:
                train.append((user, movie, rate))
        test.append((data[-1][0] - 1, data[-1][1] - 1, data[-1][2]))
        print("train", len(train))
        # print("train[0]" , len(train[0]))
        print("gbm_lr", len(test))
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):  # 评分矩阵
        train_matrix = np.zeros([self.shape[0], self.shape[1]])
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user, movie] = rating
        print("embedding shape: ", train_matrix.shape)
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        movie = []
        rate = []
        for i in data:
            user.append(i[0])
            movie.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                movie.append(j)
                rate.append(0.0)
        return np.array(user), np.array(movie), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        movie = []
        for s in testData:
            tmp_user = []
            tmp_movie = []
            u = s[0]
            m = s[1]
            tmp_user.append(u)
            tmp_movie.append(m)
            neglist = []
            neglist.append(m)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.append(j)
                tmp_user.append(u)
                tmp_movie.append(j)
            user.append(tmp_user)
            movie.append(tmp_movie)
            return [np.array(user), np.array(movie)]


class Model():
    def __init__(self, dataName="ml-1m", negNum=7, userLayer=[512, 64], itemLayer=[1024, 64], reg=1e-3, lr=0.0001,
                 maxEpochs=50, batchSize=256, earlyStop=5, checkPoint="./checkPoint", topK=10):
        self.dataName = dataName
        self.dataSet = Dataset(self.dataName)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate
        self.train = self.dataSet.train
        self.test = self.dataSet.test
        self.negNum = negNum
        self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        self.add_embedding_matrix()
        self.add_placeholders()
        self.userLayer = userLayer
        self.itemLayer = itemLayer
        self.add_model()
        self.add_loss()
        self.lr = lr
        self.add_train_step()
        self.checkPoint = checkPoint
        self.init_sess()
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.topK = topK
        self.earlyStop = earlyStop

    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)  #可输入任意shape的tensor
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding(), dtype=tf.float32)
        self.item_user_embedding = tf.transpose(self.user_item_embedding)
        # print(tf.shape(self.user_item_embedding))
        # print("self.dataSet.getEmbedding(): ", self.dataSet.getEmbedding().shape)

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user) #只是把原始的评分矩阵中的行查出来
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        # print(tf.shape(user_input))

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_w1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer) - 1):
                W = init_variable([self.userLayer[i], self.userLayer[i + 1]], "user_w" + str(i + 2))
                b = init_variable([self.userLayer[i + 1]], "user_b" + str(i + 2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))
        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_w1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer) - 1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i + 1]], "item_w" + str(i + 2))
                b = init_variable([self.itemLayer[i + 1]], "item_b" + str(i + 2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))
        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keepdims=False) / (
                    norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

    def add_loss(self):
        regRate = self.rate / self.maxRate
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        self.loss = tf.reduce_sum(losses)

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        # self.config = tf.ConfigProto()
        # self.config.gpu_options.allow_growth = True
        # self.config.allow_soft_placement = True
        # self.sess = tf.Session(config=self.config)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if os.path.exists(self.checkPoint):
            [os.remove(f) for f in os.listdir(self.checkPoint)]
        else:
            os.makedirs(self.checkPoint)

    def run(self):
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("start training")
        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch", epoch, "=" * 20)
            self.run_epoch(self.sess)
            print("=" * 50)
            print("start Evaluation")
            hr, NDCG = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
            if hr > best_hr or NDCG > best_NDCG:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - best_epoch > self.earlyStop:
                print("normal earlystop")
                break
            print("=" * 20 + "Epoch ", epoch, "End" + "=" * 20)
        print("Best HR: {}, Best_NDCG: {} at Epoch: {}".format(best_hr, best_NDCG, best_epoch))
        print("training complete")

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum) #Get negative instances
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]
        num_batches = len(train_u) // self.batchSize + 1
        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i + 1) * self.batchSize])
            train_u_batch = train_u[min_idx:max_idx]
            train_i_batch = train_i[min_idx:max_idx]
            train_r_batch = train_r[min_idx:max_idx]
            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write("\r{} / {} : loss = {}".format(i, num_batches, np.mean(losses[-verbose:])))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\n Mean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u, self.item: i, self.rate: r, self.drop: drop}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0

        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0

        hr = []
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)
            item_score_dict = {}
            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]
            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)
            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)

def main():
    classifier = Model()
    classifier.run()
    classifier.evaluate(classifier.sess, 10)

if __name__ == '__main__':
    main()

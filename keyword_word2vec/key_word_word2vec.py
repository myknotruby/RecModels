
from gensim.models import word2vec
from util import *

base_path = "/sdb1/datasets/A_CSV/"

# 保存视频明星表
df_vid_info = pd.read_csv(base_path + "vid_info.csv")
# 引入日志配置
import logging




logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
df_vid_info['key_word']= df_vid_info['key_word'].astype('str')
df_vid_info['key_word'] = df_vid_info['key_word'].apply(str_to_list)

##############
key_word_lists = df_vid_info['key_word'].values.tolist()

# import pdb; pdb.set_trace()
print(list(df_vid_info['key_word'])[:20])
key_word = []

for kw in key_word_lists:
   key_word.extend(kw)

key_word_unique_list = list(set(key_word))
print("Original #keyword:", len(key_word_unique_list))


# 引入数据集

# 切分词汇


model = word2vec.Word2Vec(key_word_lists, min_count=5, vector_size=32)

model.save(base_path + 'video_keyword.w2v.model')

model = word2vec.Word2Vec.load(base_path + 'video_keyword.w2v.model')

used_key_word = model.wv.index_to_key
print("----")
print(len(used_key_word))
print("----")
# print(model.similarity('woman', 'man'))
# print(model.wv.keys())

embeddings = []
for word in used_key_word:
    embeddings.append(model.wv[word])

df = pd.DataFrame({'key_word': used_key_word, 'embedding':embeddings}, columns=['key_word', 'embedding'])
df.to_csv(base_path + "vid_keyword_embedding.csv", index=False)


# print(model.wv[3])

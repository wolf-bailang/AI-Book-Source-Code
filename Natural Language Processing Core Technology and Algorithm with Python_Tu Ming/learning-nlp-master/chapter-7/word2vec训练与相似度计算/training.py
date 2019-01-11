# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging      # 用于输出运行日志,可以设置输出日志的等级、日志保存路径、日志文件回滚等

# %(asctime)s：打印日志的时间
# %(levelname)s：打印日志级别的名称
# %(message)s：打印日志信息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def my_function():
    wiki_news = open('./data/reduce_zhiwiki.txt', 'r')
    # sg=0 使用CBOW模型训练词向量， sg=1 使用Skip-gram模型训练词向量
    # size词向量维度
    # window表示当前词和预测词可能的最大距离，越大计算时间越长
    # min_coount最小出现次数，小于直接忽略该词
    # workers线程数
    model = Word2Vec(LineSentence(wiki_news), sg=0,size=192, window=5, min_count=5, workers=9)
    model.save('zhiwiki_news.word2vec')

if __name__ == '__main__':
    my_function()

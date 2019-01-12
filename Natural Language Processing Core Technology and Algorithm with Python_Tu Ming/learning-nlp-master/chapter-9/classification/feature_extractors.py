"""
@author: liushuchun
"""

from sklearn.feature_extraction.text import CountVectorizer        # 只考虑词汇在文本中出现的频率

# 词袋模型特征提取，词频向量化
def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)     # 计算各个词语出现的次数
    return vectorizer, features


from sklearn.feature_extraction.text import TfidfTransformer

# tfidf模型特征提取，用CountVectorizer向量化之后再调用TfidfTransformer类进行预处理
def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    # fit_transform(partData)对部分数据先拟合fit,找到该part的
    # 整体指标,如均值、方差、最大值最小值等等(根据具体转换的目的),然后对该partData进行转换
    tfidf_matrix = transformer.fit_transform(bow_matrix)

    return transformer, tfidf_matrix

# 除了考量某词汇在文本出现的频率，还关注包含这个词汇的所
# 有文本的数量能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征
from sklearn.feature_extraction.text import TfidfVectorizer     # 用于统计vectorizer中每个词语的TF-IDF值

def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features






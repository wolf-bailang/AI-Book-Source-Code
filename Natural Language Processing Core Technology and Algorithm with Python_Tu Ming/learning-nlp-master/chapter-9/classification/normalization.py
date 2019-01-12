"""
@author: liushuchun
"""
import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()

# 分词
def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens

# 去除特殊符号
def remove_special_characters(text):
    tokens = tokenize_text(text)       # 分词
    # re.compile编译正则表达式，re.escape对字符串中的非字母数字进行转义，返回一个字符串s的拷贝,不过s中的所有非
    # 字母数字字符在新串中均已被转义(在其前面添加反斜杠)。
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))      # string.punctuation包含所有标点的字符串
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# 去停用词
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text

# 归一化
def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)      # 分词
            normalized_corpus.append(text)
    return normalized_corpus

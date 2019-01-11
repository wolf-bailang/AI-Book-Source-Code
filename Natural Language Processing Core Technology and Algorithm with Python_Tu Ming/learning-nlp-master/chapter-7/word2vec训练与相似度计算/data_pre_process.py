# -*- coding: utf-8 -*-
# 中文语料的预处理

from gensim.corpora import WikiCorpus       # 维基百科处理类
import jieba
from langconv import *

def my_function():
    space = ' '
    i = 0
    l = []
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'     # 维基百科中文网页语料库
    f = open('./data/reduce_zhiwiki.txt', 'w')
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})     # 将lemmatize设置为False的主要目的是不使用pattern模块来进行英文单词的词干化处理，使用pattern会变得很慢。
    for text in wiki.get_texts():       # get_texts将维基里的每篇文章转换为1行text文本，并且去掉了标点符号等内容
        for temp_sentence in text:
            temp_sentence = Converter('zh-hans').convert(temp_sentence)     # 繁体字转简体字  简转繁zh-hant
            seg_list = list(jieba.cut(temp_sentence))
            for temp_term in seg_list:
                l.append(temp_term)
        f.write(space.join(l) + '\n')
        l = []
        i = i + 1

        if (i %200 == 0):
            print('Saved ' + str(i) + ' articles')
    f.close()

if __name__ == '__main__':
    my_function()

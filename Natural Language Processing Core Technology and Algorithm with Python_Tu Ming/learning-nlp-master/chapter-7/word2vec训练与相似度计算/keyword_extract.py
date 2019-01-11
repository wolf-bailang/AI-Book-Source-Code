# -*- coding: utf-8 -*-


import jieba.posseg as pseg
from jieba import analyse

def keyword_extract(data, file_name):
   tfidf = analyse.extract_tags
   keywords = tfidf(data)     # jieba tfidf关键词提取方法
   return keywords

# 将文档的每句话进行关键词提取并保存
def getKeywords(docpath, savepath):
   with open(docpath, 'r') as docf, open(savepath, 'w') as outf:
      for data in docf:
         data = data[:len(data)-1]
         keywords = keyword_extract(data, savepath)
         for word in keywords:
            outf.write(word + ' ')
         outf.write('\n')

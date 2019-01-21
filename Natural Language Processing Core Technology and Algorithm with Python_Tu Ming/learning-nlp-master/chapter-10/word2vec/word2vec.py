from gensim.corpora import WikiCorpus
import codecs       # 进行文件操作-读写中英文字符
import jieba
import multiprocessing      # 多进程组件
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 数据预处理
space = " "

with open('wiki-zh-article.txt', 'w', encoding="utf-8") as f:
    wiki = WikiCorpus('zhwiki-lastes-pages-articles.xml.bz2', lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        f.write(space.join(text) + "\n")
print("Finished Saved")

# 繁体字处理
opencc -i corpus.txt -o wiki-corpus.txt -c t2s,json

# 分词
infile = 'wiki-zh-article-zhs.txt'
outfile = 'wiki-zh-words.txt'

# 由于python中默认的编码是ascii，如果直接使用open方法得到文件对象然后进行文件的读写，都将无法使用包含中文字符
# rb  仅读，二进制，待打开的文件必须存在
descsFile = codecs.open(infile, 'rb', encoding='utf-8')
i = 0
with open(outfile, 'w', encoding='utf-8') as f:
    for line in descsFile:
        i += 1
        if i % 10000 == 0:
            print(i)
        line = line.strip()
        words = jieba.cut(line)
        for word in words:
            f.write(word + ' ')
        f.write('\n')

# 运行word2vec训练
inp = 'wiki-zh-words.txt'
outp1 = 'wiki-zh-model'
outp2 = 'wiki-zh-vector'

# sentences：可以是一个List，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
# size：是指输出的词的向量维数，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
# window：为训练的窗口大小，8表示每个词考虑前8个词与后8个词（实际代码中还有一个随机选窗口的过程，窗口大小<=5)，默认值为5。
# min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
# workers:参数控制训练的并行数。
model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save(outp1)       # 保存模型
# 保存的文件不能利用文本编辑器查看但是保存了训练的全部信息，可以在读取后追加训练
model.save_word2vec_format(outp2, binary=False)     # 保存词向量
# 保存为word2vec文本格式但是保存时丢失了词汇树等部分信息，不能追加训练
# model.save_word2vec_format(outp2,binary = True)

# 看效果
model = Word2Vec.load('./wiki-zh-model')
# 如果之前用文本保存，可以用下面的方法加载
# model = Word2Vec.load_word2vec_format('./wiki-zh-vector', binary=False)

res = model.most_similar('时间')
print(res)
import os
import torchtext.data as data
import sys
import config


class file_content:
    '获取train，test，val等文件内容'

    def __init__(self, name: str):
        basedir = os.path.dirname(__file__)
        filename = 'cnews/cnews.'+name+'.txt'
        self.filepath = os.path.join(basedir, filename)
        self.text, self.label = self.read_file()

    def read_file(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            text, label = [], []
            for i in f.readlines():
                tmp = i.strip().split('\t')
                text.append(tmp[1])
                label.append(tmp[0])
            return text, label


class cnews_data:
    '''构造分类的数据库'''

    def __init__(self, fix_length: int, max_size: '总单词的最大数量'):
        '''建立field'''
        self.TEXT = data.Field(
            fix_length=fix_length, batch_first=True, tokenize=lambda x: [i for i in x])
        self.LABEL = data.Field(is_target=True, pad_token=None,
                                unk_token=None, sequential=False, batch_first=True)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.max_size = max_size

    def build_ex_field(self, content):
        '建立example'
        fields = [('text', self.TEXT), ('label', self.LABEL)]
        examples = []
        for t, l in zip(content.text, content.label):
            examples.append(data.Example.fromlist([t, l], fields))
        return examples, fields

    def getDataset(self, train: file_content, val: file_content, test: file_content):
        '建立dataset，并将其传回field'
        e, f = self.build_ex_field(train)
        self.train_data = data.Dataset(e, f)
        e, f = self.build_ex_field(val)
        self.val_data = data.Dataset(e, f)
        e, f = self.build_ex_field(test)
        self.test_data = data.Dataset(e, f)

        self.TEXT.build_vocab(self.train_data, max_size=self.max_size-2)
        self.LABEL.build_vocab(self.train_data)


class cnews_dataset(cnews_data):
    def __init__(self, fix_length: int, max_size: '单词总个数'):
        super(cnews_dataset, self).__init__(fix_length, max_size)
        self.getDataset(file_content('train'), file_content(
            'test'), file_content('val'))

    def Iterator(self, batch_size) -> data.Iterator:

        return data.Iterator.splits((self.train_data, self.test_data),
                             batch_size=batch_size, device=config.device, 
                             sort_key=lambda x: x.text)


if __name__ == '__main__':
    c = cnews_dataset(fix_length=200, max_size=5000)
    print(c.TEXT.vocab.freqs.most_common(5))
    n,m = c.Iterator(batch_size=64)
    x = next(iter(n))
    print(''.join([c.TEXT.vocab.itos[i]for i in x.text[0]]))
    print(c.LABEL.vocab.itos[x.label[0]])

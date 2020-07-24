import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import avgModel
from data import cnews_dataset
import config

def predict(text:str,model:avgModel):
    model.eval()
    l = [data.TEXT.vocab.stoi[i] for i in text]
    l = torch.tensor(l).unsqueeze(0).to(config.device)
    p = model(l)
    index = p.max(0)[1]
    print(data.LABEL.vocab.itos[index])
    print(data.LABEL.vocab.itos)
    # print(data.LABEL.vocab.itos[label])


if __name__ == '__main__':
    model = avgModel(config.max_vocab_size, config.embedding_size,
                     config.dropout, config.label_size).to(config.device)
    model.load_state_dict(torch.load("../test.pt"))
    data = cnews_dataset(fix_length=config.fix_length,max_size=config.max_vocab_size)
    s = '源达信息表示，现阶段从整体市场而言，低位周期板块从估值、阶段涨幅及位置看具有一定优势，存在补涨动能，均衡配置的策略较为合理，从关注的方向看，价值周期股中关注化工、消费性建材为主，而成长方向中关注军工板块。指数层面，依然维持目前阶段震荡的观点，一方面外部扰动因素影响压制指数上行；另一方面市场中期上行的基础未变，成交量提升后，结构性行情依然存在'
    predict(s,model)

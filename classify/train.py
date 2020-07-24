import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import avgModel
from data import cnews_dataset
import config


def accuracy(preds, y):
    f_preds = preds.max(1)[1]
    correct = (f_preds == y).float()
    acc = sum(correct)/len(correct)
    return acc

# a = torch.randn(20,10)
# b = torch.randint(0,10,(20,1)).squeeze()
# a = a.max(1)[1]
# correct = (b == a).float()
# acc = sum(correct)/len(correct)

def predict(text:list,label:int,data:cnews_dataset):
    print(''.join([data.TEXT.vocab.itos[i]for i in text]))
    print(data.LABEL.vocab.itos[label])


def evaluate(model: avgModel, data, criterion: nn.CrossEntropyLoss):
    epoch_loss, epoch_acc = .0, .0
    batch_num = len(data)
    print(batch_num)
    for it in data:
        pred = model(it.text)

        loss = criterion(pred, it.label)
        # print(pred.shape,it.label.shape)
        acc = accuracy(pred, it.label)

        epoch_loss += loss
        epoch_acc += acc
        # print( f'测试集:,精准度:{acc},loss:{loss}')

    print(
        f'测试集:,精准度:{epoch_acc/batch_num},loss:{epoch_loss/batch_num}')
    return epoch_acc/batch_num


def train(epoch_num, model: avgModel, data,tdata, criterion: nn.CrossEntropyLoss, optimizer=optim.Adam):
    batch_num = len(data)
    best_acc = 0.4
    for epoch in range(epoch_num):
        epoch_loss, epoch_acc = .0, .0
        for it in data:
            pred = model(it.text)

            loss = criterion(pred, it.label)
            acc = accuracy(pred, it.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            epoch_acc += acc
        with torch.no_grad():
            model.eval()
            e_acc = evaluate(model, tdata, criterion)
            model.train()
            if(e_acc > best_acc):
                torch.save(model.state_dict(), 'test.pt')
                best_acc = e_acc
        
        print(
            f'Epoch:{epoch},精准度:{epoch_acc/batch_num},loss:{epoch_loss/batch_num}')
        


if __name__ == '__main__':
    model = avgModel(config.max_vocab_size, config.embedding_size,
                     config.dropout, config.label_size).to(config.device)
    criterion = nn.CrossEntropyLoss().to(config.device)
    optimizer = optim.Adam(model.parameters())
    data = cnews_dataset(fix_length=config.fix_length,max_size=config.max_vocab_size)
    train_data,test_data= data.Iterator(64)
    train(50,model,train_data,test_data,criterion,optimizer)

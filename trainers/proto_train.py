import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss

def default_train(train_loader,model,
    optimizer,writer,iter_counter):

    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)
    writer.add_scalar('scale',model.scale.item(),iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp,_) in enumerate(train_loader):

        iter_counter += 1

        inp = inp.cuda()
        log_prediction = model(inp)
        
        loss = criterion(log_prediction,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way

        avg_acc += acc
        avg_loss += loss_value

    avg_acc = avg_acc/(i+1)
    avg_loss = avg_loss/(i+1)

    writer.add_scalar('proto_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc

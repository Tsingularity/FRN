import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss,BCEWithLogitsLoss,BCELoss


def auxrank(support):
    way = support.size(0)
    shot = support.size(1)
    support = support/support.norm(2).unsqueeze(-1)
    L1 = torch.zeros((way**2-way)//2).long().cuda()
    L2 = torch.zeros((way**2-way)//2).long().cuda()
    counter = 0
    for i in range(way):
        for j in range(i):
            L1[counter] = i
            L2[counter] = j
            counter += 1
    s1 = support.index_select(0, L1) # (s^2-s)/2, s, d
    s2 = support.index_select(0, L2) # (s^2-s)/2, s, d
    dists = s1.matmul(s2.permute(0,2,1)) # (s^2-s)/2, s, s
    assert dists.size(-1)==shot
    frobs = dists.pow(2).sum(-1).sum(-1)
    return frobs.sum().mul(.03)



def default_train(train_loader,model,optimizer,writer,iter_counter):
    
    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()
    criterion = nn.NLLLoss().cuda()
    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)
    writer.add_scalar('scale',model.scale.item(),iter_counter)
    writer.add_scalar('alpha',model.r[0].item(),iter_counter)
    writer.add_scalar('beta',model.r[1].item(),iter_counter)

    avg_frn_loss = 0
    avg_aux_loss = 0
    avg_loss = 0
    avg_acc = 0

    for i, (inp,_) in enumerate(train_loader):

        iter_counter += 1
        inp = inp.cuda()
        log_prediction, s = model(inp)
        frn_loss = criterion(log_prediction,target)
        aux_loss = auxrank(s)
        loss = frn_loss + aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way

        avg_acc += acc
        avg_frn_loss += frn_loss.item()
        avg_aux_loss += aux_loss.item()
        avg_loss += loss.item()

    avg_acc = avg_acc/(i+1)
    avg_loss = avg_loss/(i+1)
    avg_aux_loss = avg_aux_loss/(i+1)
    avg_frn_loss = avg_frn_loss/(i+1)

    writer.add_scalar('total_loss',avg_loss,iter_counter)
    writer.add_scalar('frn_loss',avg_frn_loss,iter_counter)
    writer.add_scalar('aux_loss',avg_aux_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc


def pre_train(train_loader,model,optimizer,writer,iter_counter):

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr',lr,iter_counter)

    writer.add_scalar('scale',model.scale.item(),iter_counter)
    writer.add_scalar('alpha',model.r[0].item(),iter_counter)
    writer.add_scalar('beta',model.r[1].item(),iter_counter)
    criterion = NLLLoss().cuda()

    avg_loss = 0
    avg_acc = 0

    for i, (inp,target) in enumerate(train_loader):

        iter_counter += 1
        batch_size = target.size(0)
        target = target.cuda()

        inp = inp.cuda()
        log_prediction = model.forward_pretrain(inp)
        
        loss = criterion(log_prediction,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*(torch.sum(torch.eq(max_index,target)).float()/batch_size).item()

        avg_acc += acc
        avg_loss += loss.item()

    avg_loss = avg_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('pretrain_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc

import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm

def train(
    model,
    train_dataloader, 
    val_dataloader,
    optimizer=None, 
    criterion=None,
    vocab_tox=None,
    vocab_detox=None, 
    teacher_force=1.0,
    teacher_decay=1.0,
    teacher_epoch=1,
    lr=1e-3,
    start_epoch=1,
    epochs=15,
    model_path='model.pt'
):
    
    """
        Train the given model
            :param model: model to train
            :param train_dataloader: dataloader for train
            :param val_dataloader: dataloader for validation
            :param vocab_tox: vocabulary for toxic text
            :param vocab_detox: vocabulary for translated text
            :param optimzier: optimizer for model,     default = Adam
            :param criterion: loss function for model, default = CrossEntropyLoss
            :param lr: learning rate for optimizer and criterion, default = 1e-3
            :param start_epoch: the epoch start from
            :param epochs: number of epochs to train model: default = 15
            :param teacher_force: initial teacher force                            # used only for seq2seq
            :param teacher_decay: decay teacher_force                              # used only for seq2seq
            :param teacher_epoch: after teacher_epoch do decay                     # used only for seq2seq

        
        return:
            :loss_train_list: return loss of train through all epochs
            :loss_val_list: return loss of val though all epochs
    """
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=vocab_tox['<pad>'])
    
    total = 0
    loss_total = 0
    best = float('inf')
    loss_train_list = []
    loss_val_list = []
    for epoch in range(start_epoch, epochs + start_epoch):
        loss_train = train_epoch(epoch, train_dataloader, model, optimizer, criterion, teacher_force=teacher_force)
        best, loss_val = val_epoch(epoch, val_dataloader, model, criterion, best_so_far=best, model_path=model_path)
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)
        
        if epoch % teacher_epoch == 0:
            teacher_force *= teacher_decay
    
    return loss_train_list, loss_val_list


def train_epoch(epoch, dataloader, model, optimizer, criterion, teacher_force=None):

    total_loss = 0
    total = 0
    loop = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        desc=f"Epoch {epoch}: train",
        leave=True,
    )
    
    model.train()
    for i, batch in loop:
        input, target = batch

        optimizer.zero_grad()
        
        outputs = model(
            input, 
            target[:, :-1], 
            teacher_force
        )
        
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target[:, 1:].reshape(-1)
        )
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * input.shape[0]
        total += input.shape[0]
        loop.set_postfix({"loss": total_loss/total})

    return total_loss / total


def val_epoch(epoch, dataloader, model,
          criterion, best_so_far=0.0, model_path='model.pt'):
    
    total_loss = 0
    total = 0
    loop = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        desc=f"Epoch {epoch}: val",
        leave=True,
    )
    
    with torch.no_grad():
        model.eval()
        for i, batch in loop:
            input, target = batch

            outputs = model(
                input,
                target[:, :-1],
                0 # Without teacher force
            )
            
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target[:, 1:].reshape(-1),
            )

            total_loss += loss.item() * input.shape[0]
            total += input.shape[0]
            loop.set_postfix({"loss": total_loss/total})

        Loss = total_loss / total
        if Loss < best_so_far:
            torch.save(model, model_path)
            return Loss, total_loss / total

    return best_so_far, total_loss / total

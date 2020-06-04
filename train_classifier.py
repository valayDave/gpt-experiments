from language_model_tools import *
from data_formatting import *
import training_data_extraction
from pandas import DataFrame
from transformers import AdamW,get_linear_schedule_with_warmup
import time
import os
import click
import datetime
import json

output_dir = "./storage/models/classifier/"+str(int(time.time()))+"/"

from transformers import ( 
    WEIGHTS_NAME, 
    CONFIG_NAME,
    PreTrainedTokenizer
)   
def safe_mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except:
        pass

def save_training_params(batch_size,epochs,lr,warmup,samples,dir_path):
    saved_data = dict(
                    batch_size =batch_size,
                    epochs=epochs,
                    lr =lr,
                    warmup =warmup,
                    samples=samples,
                    date=datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
    safe_mkdir(dir_path)
    with open(os.path.join(dir_path,'params.json'),'w') as outfile:
        json.dump(saved_data,outfile)

def checkpoint_model(ml_model,tokenizer,dir_path):
    safe_mkdir(dir_path)
    output_model_file = os.path.join(dir_path, WEIGHTS_NAME)
    output_config_file = os.path.join(dir_path, CONFIG_NAME)
    torch.save(ml_model.state_dict(), output_model_file)
    ml_model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(dir_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, loss_fn, optimizer,scheduler,device, print_frequency = 2,curr_epoch=1):
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[]
    }
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(curr_epoch))

    # switch to train mode
    model.train()
    # https://github.com/pytorch/pytorch/issues/16417#issuecomment-566654504
    end = time.time()
    for i, (input_ids,attention_mask, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = torch.argmax(labels,dim=1).to(device,non_blocking=True)
        # compute output
        output = model(input_ids,attention_mask=attention_mask)

        loss = loss_fn(output, labels)

        # measure accuracy and record loss
        acc1 = accuracy(output, labels)
        # print(loss.item())
        # print(acc1)
        # print(i)
        losses.update(loss.item(), input_ids.size(0))
        top1.update(acc1[0].tolist()[0], input_ids.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        history['accuracy'].append(float(top1.avg))
        history['loss'].append(float(losses.avg))
        history['batch_time'].append(float(batch_time.avg))
        
        if i % print_frequency == 0:
            progress.display(i)
    
    return history

def validate(validation_loader, model, loss_fn,device, print_frequency = 2,curr_epoch=1):
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[]
    }
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(validation_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(curr_epoch))

    # switch to train mode
        # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # https://github.com/pytorch/pytorch/issues/16417#issuecomment-566654504
        end = time.time()
        for i, (input_ids,attention_mask, labels) in enumerate(validation_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = torch.argmax(labels,dim=1).to(device,non_blocking=True)
            # compute output
            output = model(input_ids,attention_mask=attention_mask)

            loss = loss_fn(output, labels)

            # measure accuracy and record loss
            acc1 = accuracy(output, labels)
            losses.update(loss.item(), input_ids.size(0))
            top1.update(acc1[0].tolist()[0], input_ids.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            history['accuracy'].append(float(top1.avg))
            history['loss'].append(float(losses.avg))
            history['batch_time'].append(float(batch_time.avg))
            
            if i % print_frequency == 0:
                progress.display(i)
        
    return history

def cross_entropy_one_hot(input_val, target):
    _, labels = target.max(dim=0)
    return nn.CrossEntropyLoss()(input_val, labels)

def training_loop(train_loader,val_loader,tokenizer,num_epochs, model, loss_fn, optimizer,scheduler,device, print_frequency = 2,checkpoint=True):
    print("Training/Testing Datasets Loaded!")
    epoch_histories = {
        'train': [],
        'validation': []
    }
    for epoch in range(num_epochs):
        print("Training Epoch : ",epoch)
        # train for one epoch
        train_history = train(train_loader, model, loss_fn, optimizer,scheduler ,device)
        epoch_histories['train'].append(train_history)
        # evaluate on validation set
        validation_history = validate(val_loader, model, loss_fn,device,curr_epoch=epoch)
        epoch_histories['validation'].append(validation_history)
        
        if epoch % 2 == 0 and epoch != 0 and checkpoint:
            checkpoint_model(model,tokenizer,output_dir+str(epoch))

    return epoch_histories , model


@click.command(help='Train GPT-2 Classifier on Publisher Content')
@click.option('--batch_size',default=4,type=int,help='Batch Size of Model')
@click.option('--num_epochs',default=2,type=int,help='Epoch of Model')
@click.option('--lr',default=5e-5,type=float,help='Learning Rate')
@click.option('--eps',default= 1e-8,type=float,help='epsilon')
@click.option('--warmup',default=10000,type=int,help='Warmup Steps')
@click.option('--num_samples',default=None,type=int,help='Number of Samples to Train on')
def train_classifier(lr = 5e-5,eps = 1e-8 ,batch_size = 2,warmup =100,num_epochs=3,num_samples=None):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    processor = DocumentDataPreprocessor(tokenizer)
    save_training_params(batch_size,num_epochs,lr,warmup,num_samples,output_dir)
    df = training_data_extraction.iter_one(num_samples=num_samples)
    training_content_df = df['training_content']
    labels = df['source.id']
    tensor_dataset=processor.prepare_dataset(training_content_df,labels,max_length=1024)
    model = GPT2ClassificationModel.from_pretrained('gpt2') 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.resize_token_embeddings(len(processor.tokenizer))
    optimizer = AdamW(model.parameters(),
        lr = lr, 
        eps = eps 
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=-1)

    train,validation = processor.split_dataset(tensor_dataset)
    train_dataloader = DataLoader(
        train,  # The training samples.
        sampler = RandomSampler(train), # Select batches randomly
        batch_size = batch_size # Trains with this batch size.
    )
    validation_dataloader = DataLoader(
        validation,  # The training samples.
        sampler = RandomSampler(validation), # Select batches randomly
        batch_size = batch_size # Trains with this batch size.
    )
    loss_fn = nn.CrossEntropyLoss()

    history,model = training_loop(train_dataloader,validation_dataloader,processor.tokenizer,num_epochs, model, loss_fn, optimizer,scheduler,device, print_frequency = 2)
    checkpoint_model(model,tokenizer,output_dir+str(num_epochs))
    with open(os.path.join(output_dir,'histories.json'),'w') as outfile:
        json.dump(history,outfile)

    return history,model,processor.tokenizer

if __name__=='__main__':
    train_classifier()
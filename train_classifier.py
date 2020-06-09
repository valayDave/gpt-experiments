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
from metrics import ConfusionMatrix

output_dir = "./storage/models/classifier/"+str(int(time.time()))+"/"
TRAINING_NOTE = 'Source_Extraction_Classifier'

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


def checkpoint_model(ml_model,tokenizer,dir_path):
    """checkpoint_model 
    Checkpoints the model/tokenizer etc. 
    :param processor: This is the Dataset preparer with Tokenizer
    """
    safe_mkdir(dir_path)
    output_model_file = os.path.join(dir_path, WEIGHTS_NAME)
    output_config_file = os.path.join(dir_path, CONFIG_NAME)
    torch.save(ml_model.state_dict(), output_model_file)
    ml_model.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(dir_path)

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

class HyperParamDict():
    def __init__(self,batch_size,epochs,lr,warmup,samples,column_split_order,note):
        self.batch_size = batch_size
        self.epochs= epochs
        self.lr = lr
        self.warmup = warmup
        self.samples= samples
        self.column_split_order = column_split_order
        self.note = note
    
    def to_json(self):
        return self.__json__()
    
    def __str__(self):
        return """
        Note : {note}
        
        Num Samples :  {samples}
        
        Num Epochs :  {epochs}
              
        batch_size : {batch_size}
        
        Learning Rate : {lr}
        
        WarmUp :  {warmup}

        Column Order : {column_split_order} 

        Date : {date}
        
        """.format(
                batch_size= str(self.batch_size),
                epochs= str(self.epochs),
                lr= str(self.lr),
                warmup= str(self.warmup),
                samples= str(self.samples),
                date=datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"),
                column_split_order=','.join(self.column_split_order),
                note=self.note
            )
    
    def __json__(self):
        saved_data = dict(
                    batch_size= self.batch_size,
                    epochs= self.epochs,
                    lr= self.lr,
                    warmup= self.warmup,
                    samples= self.samples,
                    column_split_order=self.column_split_order,
                    date=datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"),
                    note=self.note)
        
        return saved_data

def save_training_params(batch_size,epochs,lr,warmup,samples,dir_path,column_split_order,note=TRAINING_NOTE):
    saved_data = HyperParamDict(batch_size,epochs,lr,warmup,samples,column_split_order,note)
    safe_mkdir(dir_path)
    with open(os.path.join(dir_path,'params.json'),'w') as outfile:
        json.dump(saved_data.to_json(),outfile)
    return saved_data

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


def accuracy(output, target, topk=(1,),conf_matrix=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        #Add to confusion matrix if supplied.
        if conf_matrix is not None:
            pred_argmax = torch.argmax(pred,dim=1)
            conf_matrix.add_batch(pred_argmax.numpy(),target.numpy())

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, loss_fn, optimizer,scheduler,device, print_frequency = 2,curr_epoch=1,column_split_order=[]):
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[],
        'confusion_matrix':None,
        'classification_metrics' : None
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
    conf_matrix = None
    # if len(column_split_order) > 0:
    #     conf_matrix = ConfusionMatrix(column_split_order)
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
        acc1 = accuracy(output, labels,conf_matrix=None)
        # print(loss.item())
        # print(acc1)
        # print(i)
        losses.update(loss.item(), input_ids.size(0))
        top1.update(acc1[0].tolist()[0], input_ids.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_frequency == 0:
            progress.display(i)
    
    history['accuracy'].append(float(top1.avg))
    history['loss'].append(float(losses.avg))
    history['batch_time'].append(float(batch_time.avg))
    if conf_matrix is not None:
        history['classification_metrics'] = conf_matrix.get_all_metrics()
        history['confusion_matrix'] = str(conf_matrix)

    return history

def validate(validation_loader, model, loss_fn, device, print_frequency = 2,curr_epoch=1,column_split_order=[]):
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[],
        'classification_metrics':None,
        'confusion_matrix':None
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
    conf_matrix = None
    if len(column_split_order) > 0:
        conf_matrix = ConfusionMatrix(column_split_order)

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
            acc1 = accuracy(output, labels,conf_matrix=conf_matrix)

            losses.update(loss.item(), input_ids.size(0))
            top1.update(acc1[0].tolist()[0], input_ids.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_frequency == 0:
                progress.display(i)
        
        history['accuracy'].append(float(top1.avg))
        history['loss'].append(float(losses.avg))
        history['batch_time'].append(float(batch_time.avg))
        if conf_matrix is not None:
            history['classification_metrics'] = conf_matrix.get_all_metrics()
            history['confusion_matrix'] = str(conf_matrix)

    return history

def cross_entropy_one_hot(input_val, target):
    _, labels = target.max(dim=0)
    return nn.CrossEntropyLoss()(input_val, labels)

def training_loop(train_loader,val_loader,tokenizer,num_epochs, model, loss_fn, optimizer,scheduler,device,checkpoint_every=10, print_frequency = 2,checkpoint=True,checkpoint_dir=output_dir,column_split_order=[]):
    print("Training/Testing Datasets Loaded!")
    epoch_histories = {
        'train': [],
        'validation': []
    }
    for epoch in range(num_epochs):
        print("Training Epoch : ",epoch)
        # train for one epoch
        train_history = train(train_loader, model, loss_fn, optimizer,scheduler ,device,curr_epoch=epoch,column_split_order=column_split_order)
        epoch_histories['train'].append(train_history)
        # evaluate on validation set
        validation_history = validate(val_loader, model, loss_fn,device,curr_epoch=epoch,column_split_order=column_split_order)
        epoch_histories['validation'].append(validation_history)
        
        if epoch % checkpoint_every == 0 and epoch != 0 and checkpoint:
            checkpoint_model(model,tokenizer,checkpoint_dir+str(epoch))

    return epoch_histories , model


@click.command(help='Train GPT-2 Classifier on Publisher Content')
@click.option('--batch_size',default=4,type=int,help='Batch Size of Model')
@click.option('--num_epochs',default=2,type=int,help='Epoch of Model')
@click.option('--lr',default=5e-5,type=float,help='Learning Rate')
@click.option('--eps',default= 1e-8,type=float,help='epsilon')
@click.option('--warmup',default=1000,type=int,help='Warmup Steps')
@click.option('--checkpoint_every',default=10,type=int,help='Checkpoint Every Steps')
@click.option('--num_samples',default=None,type=int,help='Number of Samples to Train on')
@click.option('--train_split',default=0.7,type=float,help='Split of Training/Validation')
@click.option('--gradient_accumulation_steps',default=1,type=int,help="Number of Steps for Grad Accumilation for Linear Scheduler")
def train_classifier(lr = 5e-5,eps = 1e-8 ,batch_size = 2,warmup =100,num_epochs=3,num_samples=None,train_split=None,checkpoint_every=None,gradient_accumulation_steps=1):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    processor = DocumentDataPreprocessor(tokenizer)
    df = training_data_extraction.iter_one(num_samples=num_samples)
    if num_samples is None:
        num_samples = len(df)
    training_content_df = df['training_content']
    labels = df['source.id']
    tensor_dataset , column_split_order =processor.prepare_dataset(training_content_df,labels,max_length=1024)
    train_params = save_training_params(batch_size,num_epochs,lr,warmup,num_samples,output_dir,column_split_order)
    print(str(train_params))
    model = GPT2ClassificationModel.from_pretrained('gpt2',num_output_labels=len(column_split_order)) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.resize_token_embeddings(len(processor.tokenizer))
    optimizer = AdamW(model.parameters(),
        lr = lr, 
        eps = eps 
    )
    train,validation = processor.split_dataset(tensor_dataset,train_percent=train_split)
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
    
    
    t_total = ( # Derived From ;https://github.com/huggingface/transformers/blob/f9414f7553d3f1872b372990ef03205c0d1141df/examples/lightning_base.py#L114
        (len(train_dataloader) // (batch_size))
        // gradient_accumulation_steps
        * float(num_epochs)
    )

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, t_total)

    loss_fn = nn.CrossEntropyLoss()

    history , model = training_loop(
        train_dataloader,\
        validation_dataloader,\
        processor.tokenizer,\
        num_epochs,\
        model,\
        loss_fn,\
        optimizer,\
        scheduler,\
        device,\
        checkpoint_every=checkpoint_every,\
        print_frequency = 2,\
        column_split_order=column_split_order
    )
    checkpoint_model(model,processor.tokenizer,output_dir+str(num_epochs))
    with open(os.path.join(output_dir,'histories.json'),'w') as outfile:
        json.dump(history,outfile)

    return history,model,processor

if __name__=='__main__':
    train_classifier()
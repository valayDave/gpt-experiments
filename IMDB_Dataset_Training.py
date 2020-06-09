import torchtext
from torchtext.datasets import IMDB
import pandas
from torch.utils.data import DataLoader,RandomSampler
from language_model_tools import IMDBSentimentSytleFormater,GPT2Tokenizer
from language_model_tools import DocumentDataPreprocessor,GPT2ClassificationModel,nn
import train_classifier as ClassifierTrainer
from transformers import AdamW,get_linear_schedule_with_warmup
import time
import click
import torch
import os
import json


output_dir = "./storage/models/imdb-classifier/"+str(int(time.time()))+"/"
TRAINING_NOTE = 'IMDB_Sentiment_Classification'
STARTING_MESSAGE = '''
Starting Training for {note}
'''.format(note=TRAINING_NOTE)

def get_data_loader(
        doc_processor:DocumentDataPreprocessor,\
        batch_size=3,\
        dataset_path ='data/IMDB/aclImdb/train',
        MAX_WORD_COUNT=1000,\
        MIN_DOC_THRESHOLD=300,\
        MIN_WORD_COUNT=0,
        num_samples=None
    ):
    text_preprocessing = None #lambda x:mdl.model_processor(x)
    label_preprocessing = None # lambda x:1 if 'pos' else 0
    TEXT = torchtext.data.RawField(preprocessing=text_preprocessing)
    LABEL = torchtext.data.RawField(is_target=True,preprocessing=label_preprocessing)
    dataset = IMDB(dataset_path,text_field=TEXT,label_field=LABEL)
    data_objects = [{'text':i.text,'label':i.label} for i in dataset.examples]
    df = pandas.DataFrame(data_objects)
    df['training_content'] =  df.apply(lambda row: doc_processor.formatter(row['text']),axis=1)
    df = df[df['training_content'].str.split().str.len() <= MAX_WORD_COUNT]
    # Filtering post cleanup.
    df = df[df['training_content'].str.split().str.len() >= MIN_WORD_COUNT]
    if num_samples is not None:
        df = df.sample(n=num_samples)
    labels = df['label']
    training_content_df = df['training_content']
    tensor_dataset , column_split_order =doc_processor.prepare_dataset(training_content_df,labels,max_length=1024)
    dataloader = DataLoader(
        tensor_dataset,  # The training samples.
        sampler = RandomSampler(tensor_dataset), # Select batches randomly
        batch_size = batch_size # Trains with this batch size.
    )
    return dataloader,column_split_order


@click.command(help='Train GPT-2 Classifier on IMDB Review Sentiment')
@click.argument('dataset_root',default='data/IMDB/aclImdb',type=click.Path(exists=True))
@click.option('--batch_size',default=4,type=int,help='Batch Size of Model')
@click.option('--num_epochs',default=2,type=int,help='Epoch of Model')
@click.option('--lr',default=5e-5,type=float,help='Learning Rate')
@click.option('--eps',default= 1e-8,type=float,help='epsilon')
@click.option('--warmup',default=1000,type=int,help='Warmup Steps')
@click.option('--checkpoint_every',default=10,type=int,help='Checkpoint Every Steps')
@click.option('--num_samples',default=None,type=int,help='Number of Samples to Train on')
@click.option('--gradient_accumulation_steps',default=1,type=int,help="Number of Steps for Grad Accumilation for Linear Scheduler")
def train_classifier(dataset_root,lr = 5e-5,eps = 1e-8 ,batch_size = 2,warmup =100,num_epochs=3,num_samples=None,train_split=None,checkpoint_every=None,gradient_accumulation_steps=1):
    print(STARTING_MESSAGE)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    processor = DocumentDataPreprocessor(tokenizer,formatter=IMDBSentimentSytleFormater())
    train_dataset_path = os.path.join(dataset_root,'train')
    test_dataset_path = os.path.join(dataset_root,'test')
    train_dataloader, column_split_order = get_data_loader(processor,batch_size=batch_size,dataset_path=train_dataset_path,num_samples=num_samples)
    validation_dataloader, column_split_order = get_data_loader(processor,batch_size=batch_size,dataset_path=test_dataset_path,num_samples=num_samples)
    if num_samples is None:
        num_samples = len(train_dataloader)*train_dataloader.batch_size
    train_params = ClassifierTrainer.save_training_params(batch_size,num_epochs,lr,warmup,num_samples,output_dir,column_split_order,note=TRAINING_NOTE)
    print(str(train_params))

    model = GPT2ClassificationModel.from_pretrained('gpt2',num_output_labels=2) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.resize_token_embeddings(len(processor.tokenizer))
    optimizer = AdamW(model.parameters(),
        lr = lr, 
        eps = eps 
    )
    
    t_total = ( # Derived From ;https://github.com/huggingface/transformers/blob/f9414f7553d3f1872b372990ef03205c0d1141df/examples/lightning_base.py#L114
        (len(train_dataloader) // (batch_size))
        // gradient_accumulation_steps
        * float(num_epochs)
    )

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, t_total)

    loss_fn = nn.CrossEntropyLoss()

    history , model = ClassifierTrainer.training_loop(
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
        print_frequency = 2,
        checkpoint_dir= output_dir,
        column_split_order=column_split_order
    )
    ClassifierTrainer.checkpoint_model(model,processor.tokenizer,output_dir+str(num_epochs))
    with open(os.path.join(output_dir,'histories.json'),'w') as outfile:
        json.dump(history,outfile)

    return history,model,processor

if __name__=='__main__':
    train_classifier()
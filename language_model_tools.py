import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import (
    random_split,
    DataLoader,
    RandomSampler,
    Subset,
    TensorDataset
)
from data_formatting import *
from transformers import (
    GPT2Tokenizer, 
    GPT2Model,
    PreTrainedModel,
    GPT2PreTrainedModel
)


class DocumentDataPreprocessor():
    """DocumentDataPreprocessor 
    This will manage the tokenization of documents that are formatted Using the 
    SourceTextStyleFormater. It will also prepare the XY dataset with attention mask
    
    This ensures that Tokens are set correctly set and preprocessinig can happen in different styles. 
    """
    CLASS_TOKEN = '[CLS]'

    SPECIAL_TOKENS = []

    def __init__(self,tokenizer:GPT2Tokenizer,\
                formatter=SourceTextStyleFormater()):

        self.tokenizer = tokenizer
        special_tokens_dict = {'cls_token': self.CLASS_TOKEN,'pad_token':self.tokenizer.eos_token}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.formatter = formatter
        tokens = self.formatter.get_all_tokens()
        self.tokenizer.add_tokens(tokens)
        

    def prepare_dataset(self,documents,labels,max_length=100)->TensorDataset:
        """prepare_dataset 
        The function will create the dataset into text,attention_mask,label
        """
        documents = documents
        # One-hot label conversion
        labels = labels.str.get_dummies().values.tolist()

        attention_mask = []
        input_ids = []
        # From https://colab.research.google.com/drive/13ErkLg5FZHIbnUGZRkKlL-9WNCNQPIow
        # For every Document...
        # No need for explicit attention mask extraction in GPT2
        # https://github.com/huggingface/transformers/issues/808#issuecomment-522932583
        for document in documents:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            encoded_dict = self.tokenizer.encode_plus(
                                document,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_length,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask=True,
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])
            # And its attention mask (simply differentiates padding from non-padding).

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids,dim=0)
        attention_mask = torch.cat(attention_mask,dim=0)
        labels = torch.tensor(labels)
        return TensorDataset(input_ids, attention_mask,labels)

    @staticmethod
    def split_dataset(dataset,train_percent=0.9):
        # Create a split in train-validation 
        # Calculate the number of samples to include in each set.
        if train_percent > 1:
            raise Exception('Training Percentage cannot be > 1')
        train_size = int(train_percent * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset,val_dataset


class GPT2ClassificationModel(GPT2PreTrainedModel):
    # https://huggingface.co/transformers/model_doc/gpt2.html#gpt2model
    def __init__(self, config,num_output_labels = 4):
        config.output_attentions=True
        super(GPT2ClassificationModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.CNN_Max = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.CNN_Avg = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.ff_layers = nn.Sequential(
            nn.Linear(256,10),
            nn.Linear(10,num_output_labels)
        )
        self.final_softmax = nn.Softmax(dim=1)
        self.init_weights()

    def forward(self,input_ids, position_ids=None, token_type_ids=None, lm_labels=None, attention_mask=None, past=None):
        transformer_op = self.transformer(input_ids,past=past, position_ids=position_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        hidden,past,attentions = transformer_op
        hidden = hidden.unsqueeze(1)
        # Convolve the hidden vectors
        max_op = self.CNN_Max(hidden)
        avg_op = self.CNN_Avg(hidden)
        # Shift dimensions to basically do a matrix Transpose of Max/avg pooliing from CNN
        max_op = max_op.view(max_op.size(0),max_op.size(2),-1)
        avg_op = avg_op.view(avg_op.size(0),avg_op.size(2),-1)
        # Multiple the matrixes. 
        result_op = torch.mul(avg_op,max_op)
        # average of final result.
        result_op = result_op.mean(-1)
        # final softmax for result logits. 
        result_logits = self.final_softmax(self.ff_layers(result_op))
        
        return result_logits


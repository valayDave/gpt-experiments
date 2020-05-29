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
    def __init__(self, config,):
        super(GPT2ClassificationModel, self).__init__(config)
        self.transformer = GPT2Model(config)

        self.init_weights()

    def forward(self,input_ids, position_ids=None, token_type_ids=None, lm_labels=None, attention_mask=None, past=None):
        transformer_op = self.transformer(input_ids,past=past, position_ids=position_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)


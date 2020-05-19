from torch.utils.data import Dataset, DataLoader 
import torch 
from torch.optim import AdamW 
from torch.utils.data import RandomSampler, SequentialSampler 
import numpy as np  
import os  
import random 
from transformers import ( 
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    CONFIG_MAPPING, 
    MODEL_WITH_LM_HEAD_MAPPING, 
    AutoConfig, 
    AutoModelWithLMHead, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling, 
    get_linear_schedule_with_warmup, 
    HfArgumentParser, 
    LineByLineTextDataset, 
    PreTrainedTokenizer, 
    TextDataset, 
    Trainer, 
    TrainingArguments, 
    set_seed, 
    WEIGHTS_NAME, 
    CONFIG_NAME
)   
import time
output_dir = "./storage/models/"+str(int(time.time()))+"/"
def safe_mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except:
        pass

def checkpoint_model(ml_model,dir_path):
    safe_mkdir(dir_path)
    output_model_file = os.path.join(dir_path, WEIGHTS_NAME)
    output_config_file = os.path.join(dir_path, CONFIG_NAME)
    torch.save(ml_model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(dir_path)

class ProgrammingLanguagesDataset(Dataset):
    def __init__(self,\
        filenames_path='py150_files/python100k_train.txt',\
        file_source_path='py150_files/',
        samples=7,
        block_size=512,
        tokenizer=PreTrainedTokenizer()):
        with open(filenames_path) as f: 
            data = f.read()  
        python_files = data.split('\n')
        python_files = random.sample(python_files,samples) 
        saved_data = [] 
        self.examples = []
        for file in python_files: 
            if file == '': 
                continue 
            with open(file_source_path+file,'rb') as f: 
                saved_data.append(str(f.read())) 
        text = '\n'.join(saved_data)
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i : i + block_size]
                )
            )
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
model = GPT2LMHeadModel.from_pretrained('gpt2') 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = model.to(device) 

BATCH_SIZE = 20
EPOCHS = 10
LEARNING_RATE = 0.00002
WARMUP_STEPS = 10000

model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)
script_count = 0
sum_loss = 0.0
batch_count = 0

script_loader = DataLoader(ProgrammingLanguagesDataset(tokenizer=tokenizer))
for epoch in range(EPOCHS):
    if epoch % 2 == 0 and epoch != 0:
        checkpoint_model(model,output_dir+str(epoch))
    
    print("Starting Epoch : ",epoch)
    for _ , script in enumerate(script_loader):
        outputs = model(script.to(device), labels=script.to(device))
        loss, logits = outputs[:2]                        
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
                       
        script_count = script_count + 1
        if script_count == BATCH_SIZE:
            script_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()
            
        if batch_count == 200:
            model.eval()
            print(f"sum loss {sum_loss}")
            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length =200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )

            print("Output:\n" + 100 * '-')
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            batch_count = 0
            sum_loss = 0.0
            model.train()

checkpoint_model(model,output_dir+str(EPOCHS))
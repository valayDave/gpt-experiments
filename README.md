
# GPT-2 Experiements

# Project Details

1. Scraping and Content Extraction via content crawler. 
2. GPT-2 Based classification of website content. 
3. For pretraining on Python Code Checkout : 
  http://files.srl.inf.ethz.ch/data/py150_files.tar.gz # Python Code 

  http://files.srl.inf.ethz.ch/data/py150.tar.gz # Parse Tree Representations


# Training Classifier 

nohup python -u ./train_classifier.py --batch_size 4 --num_epochs 10 --warmup 100 &
nohup python -u ./train_classifier.py --batch_size 2 --num_epochs 8 --warmup 5000 --checkpoint_every 3 &

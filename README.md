http://files.srl.inf.ethz.ch/data/py150_files.tar.gz # Python Code 

http://files.srl.inf.ethz.ch/data/py150.tar.gz # Parse Tree Representations

# Some Ideas to Try Out. 

1. GPT-2 Based classification of website content. 

2. GPT-2 Based generative models for content prediction

https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/

# Training Classifier 

nohup python -u ./train_classifier.py --batch_size 4 --num_epochs 10 --warmup 100 &
nohup python -u ./train_classifier.py --batch_size 2 --num_epochs 8 --warmup 5000 --checkpoint_every 3 &
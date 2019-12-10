# Char-level language modelling on Shakespeare
Yet another char language model inspired by [Andrej Karpathy's blogpost](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
Written on pytorch 1.2.0, but I imagine it would also work fine on 1.1.0 and 1.3.0.

# How to run
### Obtaining the data
Run this simple bash script to download the data from original repository.
```bash
sh get_data.sh
```
Alternatively, if your system does not support `wget`, you can download it by using the link in `get_data.sh`.
### Training the model
To train model simply write:
```
python main.py train
usage: main.py train [-h] [--model-name MODEL_NAME] [--vocab-name VOCAB_NAME]
                     [--use-gru] [--max-length MAX_LENGTH]
                     [--batch-size BATCH_SIZE] [--charemb-dim CHAREMB_DIM]
                     [--hidden-dim HIDDEN_DIM] [--num-layers NUM_LAYERS]
                     [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Name of the model for saving. Suffix `.pt` will be
                        added, as it will be a serialized pytorch weights
                        file.
  --vocab-name VOCAB_NAME
                        Name of the vocab for saving. Suffix `.pickle` will be
                        added, as it will be a pickled object.
  --use-gru             boolean, set it if you want to use GRU. By default the
                        model uses LSTM.
  --max-length MAX_LENGTH
                        Length of sample in batch.
  --batch-size BATCH_SIZE
                        Batch size for training.
  --charemb-dim CHAREMB_DIM
                        Embedding dimension size for characters.
  --hidden-dim HIDDEN_DIM
                        Hidden dimension size for RNN.
  --num-layers NUM_LAYERS
                        Number of stacked layers in RNN.
  --epochs EPOCHS       Number of epochs for training.

```
For example, if you want to replicate a model from [similiar tensorflow tutorial](https://www.tensorflow.org/tutorials/text/text_generation)
run `main.py` with the following arguments.
```
python main.py train --model-name gru --use-gru --max-length 100 --batch-size 64 --charemb-dim 256 --hidden-dim 1024 --num-layers 1 --epochs 10
```
Training such a model on GTX1080 takes 60-70 seconds.
### Generating text
To generate text from your trained model use:
```
usage: main.py generate [-h] [--model-name MODEL_NAME]
                        [--vocab-name VOCAB_NAME] [--use-gru]
                        [--length LENGTH] [--temperature TEMPERATURE]

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Model name to load from.
  --vocab-name VOCAB_NAME
                        Vocab name to load from.
  --use-gru             Set this flag, if model uses GRU instead of LSTM.
  --length LENGTH       Length of text to be generated.
  --temperature TEMPERATURE
                        Divisor before softmax. The closer to 0, the more
                        confident and conservativethe model will be, the
                        bigger the more random outputs will be produced.
```
For example, to run GRU model from previous paragraph write:
```
python main.py generate --model-name gru --use-gru --length 1000
```
### Output example
Generated from single letter Q after 30 epochs.
```
QUEEN MARGARET:
Hold, tell me, 'tis not long before weap,
Now may disperse him, Signior Gremio.
How far off sir! hence, madam; besides, you wrong yourself
Had never for the Lord Northumberland.
So it promise, and she'tle thou shalt not;
For, welcome, lady, have I held a foow
To see your victories, and you speak fair I shall.

PRINCE:
Reconverse on persons with such violent rest!
For nothing can say, it was, sure, which I bade them speak.
But, come, I shall raise Mistann'd my several.

Second Citizen:
To you so that lead me o'ercharged me would good:
How long sitenants upon me! Many
Miss it to vitizen there: where then,
A curseiving not, madam, let's resign
The venials world that women are draw;
When I was rash'd thus and my face, and not were never
May in his better for his hands.

CAPULET:
Poor gentleman, thy valour up to hear you to;
A sent hand to nose chined. Thus his mother!
Thy son of death doth kind us to off the Volsces,--

SIR STEPHEN SCROOP:
Men often of the nipper about him:
```

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
```
> ROMEO:
ROMEO:
Stay like a vaulina, my cousin' comes?
But is that caured to the soseless earth;
And let me hurrous wretched false than often.

ROMEO:
Of, will he know both; a happy more;
If you read to sproat ye grow; strike, I
will name to say to the queen is us
To pluck on eyes for life.

FLORIZEL:
Ene more bound?

LUCIO:
Let us be abislate e'er I,'t much a coadmate,
That I apt denied my father;
If I see thou and good country, pearinman.

LUCIO:
You will my sound Murdy royally next.


> Hello
Hellows-shrink, Montag buried:
I will move his grace from their country;
And, like to lose the sun: our sheep taad
Oe this officer of Tunin?
Shall I, exarce be the merry weeks,
O, by my old and prophece, that nows are ponted.
When it is gone, in this hand, I princely, Bexasting,
This further clamorous sister promises.
Suppose York drust vantage; or as
I will be fly on vost, fairs to me
hop from arm with their fardy pinch'd cut,
Twanting plead, all will yield my tributes unto,
If you shall Roman Duke of Hereford, 'sTir!'
These execution and my meaning crowns,
It shall the dignicities all.
O love?

POMPEY:
I reverend, you shall change me in disgrable;
This standing that filling else they rest.
God sands us that wo her dam is learness;
Throng of this contrary quickly clapt,
Thave my waters and grace O heart-goo-stap,
When the love came mark the woes from bild
That receipt Laurence' and to make our sweet
Where less in't a farewellerously, alas,
Lest flatter to live, choice,
```

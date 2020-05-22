UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"
PAD_TOKEN = "<PAD>"


data_pickle_path = "data/quora/processed/"
raw_data_file = "data/quora/questions.csv"

max_length = 26  # max length of a question in number of words
tokenization_method = "nltk"  # tokenization method
word_count_threshold = 3  # only words that occur more than this number of times will be put in vocab

# Embedding
emb_hid_dim = 256
emb_dim = 512
# Encoder
encoder_dropout = 0.5
encoder_rnn_units = 512
encoder_output_dim = 512
# Decoder
decoder_rnn_units = 512
decoder_dropout = 0.5

# hyperparams
lr = 0.001
batch_size = 150

import torch
from functools import partial
from torch.utils.data import DataLoader

from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)

def get_english_tokenizer():
    # Create a tokenizer for English using the basic English tokenizer in torchtext
    tokenizer = get_tokenizer("basic_english")
    return tokenizer

def get_data_iterator(ds_name, ds_type, data_dir):
    # Load data from WikiText2 or WikiText103 dataset based on the given name and type (train, test, valid)
    if ds_name =='WikiText2':
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == 'WikiText103':
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose from dataset: WikiText2, WikiText103")
    return data_iter

def build_vocab(data_iter, tokenizer):
    # Build a vocabulary from an iterator and the tokenizer
    # Includes setting a minimum frequency for words to be included
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq= MIN_WORD_FREQUENCY,
        )
    # Sets default index for unknown words
    vocab.set_default_index(vocab['<unk>'])
    return vocab 

def collate_cbow(batch, text_pipeline):
    # Function to collate data into batches for the CBOW model
    # Prepares input and output pairs based on the context window size
    batch_input, batch_output = [], []
    for text in batch:
        text_token_ids = text_pipeline(text)
        if len(text_token_ids) < CBOW_N_WORDS * 2 + 1:
            continue
        if MAX_SEQUENCE_LENGTH:
            text_token_ids = text_token_ids[:MAX_SEQUENCE_LENGTH]
        for idx in range(len(text_token_ids) - CBOW_N_WORDS*2):
            token_id_seq = text_token_ids[idx:(idx + CBOW_N_WORDS*2+1)]
            output = token_id_seq.pop(CBOW_N_WORDS)
            input_ = token_id_seq
            batch_input.append(input_)
            batch_output.append(output)
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def collate_skipgram(batch, text_pipeline):
    # Function to collate data into batches for the Skip-gram model
    # Prepares input and output pairs for each word and its context
    batch_input, batch_output = [], []
    for text in batch:
        text_token_ids = text_pipeline(text)
        if len(text_token_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue
        if MAX_SEQUENCE_LENGTH:
            text_token_ids = text_token_ids[:MAX_SEQUENCE_LENGTH]
        for idx in range(len(text_token_ids) - SKIPGRAM_N_WORDS*2):
            token_id_seq = text_token_ids[idx:(idx + SKIPGRAM_N_WORDS*2+1)]
            input_ = token_id_seq.pop(SKIPGRAM_N_WORDS)
            output = token_id_seq
            for o in output:
                batch_input.append(input_)
                batch_output.append(o)
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def get_dataloader_and_vocab(
    model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None
):
    # Creates a DataLoader for the specified model (CBOW or Skip-gram)
    # Optionally builds vocabulary if not provided
    data_iter = get_data_iterator(ds_name, ds_type, data_dir)
    tokenizer = get_english_tokenizer()
    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)
    text_pipeline = lambda x: vocab(tokenizer(x))  # Convert text to token numbers
    if model_name == 'cbow':
        collate_fn = collate_cbow
    elif model_name == 'skipgram':
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose from model cbow or skipgram")
    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn = partial(collate_fn, text_pipeline=text_pipeline)
    )
    return dataloader, vocab
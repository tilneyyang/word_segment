import os
import re
import numpy as np
import sys

UNK_TOKEN = '_UNK_TOKEN'
UNK_TOKEN_ID = 0


def _generate_ibo2_tag(token):
    token_len = len(token)
    assert token_len > 0
    tags = ['B']
    for i in xrange(token_len - 1):
        tags.append('I')
    return tags


def create_vocabulary(path_to_data_file, vocab_dir, vocab_size=0):
    """
    create vocabulary from data file
    :param path_to_data_file:
    :param vocab_dir: dir to store output vocabulary
    :param vocab_size: store at most vocab_size most frequent chars
    :return:
        path to the vocab file
    """
    vocab_file = os.path.join(vocab_dir, 'vocab_%s' % vocab_size)
    if os.path.exists(vocab_file):
        sys.stderr.write('vocab file %s exists! we will use the exist one.\n' % vocab_file)
        return vocab_file
    unichars = {}
    with open(path_to_data_file) as iff:
        for line in iff:
            line = line.decode('utf8').strip()
            for char in line:
                if char not in unichars:
                    unichars[char] = 0
                unichars[char] += 1
    sorted_chars = sorted(unichars.items(), key=lambda x: x[1], reverse=True)
    if vocab_size > 0:
        sorted_chars = sorted_chars[:vocab_size]
    vocab = [UNK_TOKEN]
    for char, _ in sorted_chars:
        if len(char.strip()) > 0:
            vocab.append(char)
    vocab = dict(enumerate(vocab))
    print vocab
    with open(vocab_file, 'w') as vf:
        for item in vocab.items():
            vf.write('%s\t%s\n' % (item[0], item[1].encode('utf8')))
    return vocab_file


def read_vocabulary(path_to_vocab):
    if not os.path.exists(path_to_vocab):
        raise ValueError('vocab file %s not exists! please run create_vocabulary() first.' % path_to_vocab)
    with open(path_to_vocab) as vf:
        vocab = {}
        for line in vf:
            line = line.decode('utf8').strip()
            chars_and_ids = re.split(ur'\s', line)
            assert len(chars_and_ids) == 2 or len(chars_and_ids) == 0
            vocab[chars_and_ids[1]] = int(chars_and_ids[0])
        return vocab


def read_data(path_to_data_file, strip_chars='',  maximum_size=0, encoding_func=_generate_ibo2_tag):
    """
    read training or testing data, and generate tag for them. The data should contain one sentence per line,
    and tokens of the sentence is separated by space or tab (ur'\s+').
    :param path_to_data_file:
    :param encoding_func:function to generate tags
    :param strip_chars: characters will be stripped from start and end of the sentence
    :return:
        a list contains of all unicode encoding sentences characters to their tags
    """
    data = []
    count = 0
    with open(path_to_data_file) as ipf:
        for line in ipf:
            line = line.strip().decode('utf8')
            count += 1
            if strip_chars is not None and len(strip_chars) > 0:
                if not isinstance(strip_chars, unicode):
                    strip_chars = strip_chars.decode('utf8')
                line = line.strip(strip_chars)
            tokens = re.split(ur'\s+', line)
            all_tags = []
            # FIXME transform char to char ids
            all_token_chars = []
            for token in tokens:
                all_token_chars.extend(token)
                tags = encoding_func(token)
                all_tags.extend(tags)
                print token, tags
            # we have one tags per token char
            assert len(all_tags) == len(all_token_chars)
            data.append([all_token_chars, all_tags])
            if 0 < maximum_size < count:
                break
    return data


def data_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw data.

    This generates batch_size pointers into the raw data, and allows
    minibatch iteration along these pointers.

    Args:
      raw_data: one of the raw data outputs from raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.

    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.

    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    # FIXME: fix bugs, we need to set fixed sentence length without buckets
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

if __name__ == '__main__':
    vocab_path = create_vocabulary('test', './')
    print read_vocabulary(vocab_path)

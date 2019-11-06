#!/usr/bin/env python

from __future__ import unicode_literals
import argparse
import collections
import io
import re
import progressbar
import tools

split_pattern = re.compile(r'([.,!?"\':;)(])。、')

def split_sentence(s):
    s = s.replace('\u2019', '\'')
    words = []
    for word in s.strip().split():
        words.extend(split_pattern.split(word))
    words = [w for w in words if w]
    return words

def read_file(path):
    n_lines = tools.count_lines(path)
    bar = progressbar.ProgressBar()
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for line in bar(f, max_value=n_lines):
            words = split_sentence(line)
            yield words

# # #   main function   # # #
def proc_dataset(path, vocab_path=None, vocab_size=None):
    sen_count = 0
    token_count = 0
    counts = collections.Counter()
    for words in read_file(path):
        sen_count += 1
        line = ' '.join(words)
        if vocab_path:
            for word in words:
                counts[word] += 1
        token_count += len(words)
    print('>> number of sentences: %d' % sen_count)
    print('>> number of tokens: %d' % token_count) # number of all words

    if vocab_path and vocab_size:
        vocab = [word for (word, _) in counts.most_common(vocab_size)]
        with io.open(vocab_path, 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write(word)
                f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-it", '--input-text', help='input text dataset')
    parser.add_argument(
        "-ov", '--output-vocab', help='output vocabulary file to save')
    parser.add_argument(
        "-vs", '--vocab-size', type=int, default=50000,
        help='size of vocabulary file')
    args = parser.parse_args()

    proc_dataset(
        args.input_text, vocab_path=args.output_vocab,
        vocab_size=args.vocab_size)

if __name__ == '__main__':
    main()
    
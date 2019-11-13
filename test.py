import numpy as np
import argparse
# import seq2seq 
import seq2seq_v2 

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx
import tenkey_filter_V2 as tf2

UNK = 0
EOS = 1


def load_data_file(path):
    print('loading...: %s' % path)
    with open(path, encoding="utf-8") as f:
        data = f.read()
        f.close()
        result = data.split("\n")        
    return result

def load_data_from_str(vocabulary, line):
    data = []
    words = line.strip().split()
    array = np.array([vocabulary.get(w, UNK) for w in words], np.int32)
    # 単語に対するIDを取り出している
    data.append(array)
    return data

def exchange_input(text):
    text = text.replace('5', ' ')
    text = text.replace('6', '.')
    return text

def check_quit(text):
    if text == "quit":
        print(">> Exit the program.")
        print(">> bye.")
        exit()

def pre_process(test_sentence):
    test_sentence = test_sentence.lower()
    check_quit(test_sentence)
    test_sentence = exchange_input(test_sentence)
    test_sentence = tf2.KeyinputFilter.alphab2num(test_sentence)
    return test_sentence

def main():
    parser = argparse.ArgumentParser(description='test program: test')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--testset', '-t', type=str, help='text data for test of model')
    parser.add_argument('--answerset', '-a', type=str, help='text data of answer')
    parser.add_argument('--resume', '-r', type=str, help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024, help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3, help='number of layers')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    # parser.add_argument('--out', '-o', default='result', help='directory to output the result')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load vocabulary file
    source_ids = seq2seq_v2.load_vocabulary(args.SOURCE_VOCAB)
    target_ids = seq2seq_v2.load_vocabulary(args.TARGET_VOCAB)

    # Set the current device
    device = chainer.get_device(args.device)
    device.use()

    # Setup model
    print("==== model loading ... ====")
    model = seq2seq_v2.Seq2seq(args.layer, source_ids, target_ids, args.unit)
    model.to_device(device)

    # replace keys and values
    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    if args.resume is not None:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, model, "updater/model:main/")

    # #######################################
    # # interpriter version
    # #######################################
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print("$$$  input ... alphabets or      $$$")
    # print("$$$           number sequence    $$$")
    # print("$$$  '5'...space   '6'...period  $$$")
    # print("$$$                              $$$")
    # print("$$$  if you want to exit,        $$$")
    # print("$$$      please input \"quit\"     $$$")
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # while True:
    #     print(">> please input test sentence.")
    #     test_sentence = input()
    #     test_sentence = pre_process(test_sentence)
    #     test_source = load_data_from_str(source_ids, test_sentence)
    #     result = model.translate([model.xp.array(test_source[0])])[0]     
    #     result_sentence = ' '.join([target_words[y] for y in result])
    #     print("--------------------------------------")
    #     print('# source : ' + test_sentence)
    #     print('# result : ' + result_sentence)
    #     print("--------------------------------------")


    ##########################################
    # file writing version
    ##########################################
    if args.testset is not None:
        test_data = load_data_file(args.testset)
        answer_data = load_data_file(args.answerset)
        for i in range(len(test_data)):
            test_sentence = pre_process(test_data[i])
            test_source = load_data_from_str(source_ids, test_sentence)
            result = model.translate([model.xp.array(test_source[0])])[0]        
            result_sentence = ' '.join([target_words[y] for y in result])

            print("------------------ {} --------------------".format(i+1))
            print('# source : ' + test_sentence)
            print('# result : ' + result_sentence)
            print("# answer : " + answer_data[i])

if __name__ == '__main__':
    main()

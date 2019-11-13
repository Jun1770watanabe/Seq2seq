import re
import io
import argparse
import numpy as np
import tools
from tqdm import tqdm
import progressbar


class KeyinputFilter:
    """ 疑似的な入力を作成する際に利用するフィルタ
    1 ローマ字キー入力文字列を数字列に置き換えるフィルタ
    2 かな文字列をローマ字キー入力文字列に置き換えるフィルタ
    """

    _KEYIN2NUMBER_TABLE = str.maketrans(
        '1qaz2wsx3edc4rfv5tgb6yhn7ujm8ik9ol0p',
        '111122223333444444447777777788899900')
    _NUMBER_ZEN2HAN_TABLE = str.maketrans(
        '０１２３４５６７８９．。，、／　￥＊！？；：＠＆＝',
        '0123456789..,,/ \*!?;:@&=')
    _ALPHABET_ZEN2HAN_TABLE = str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    _KANA2CHAR1 = {
        'ア': 'a',   'イ': 'i',   'ウ': 'u',   'エ': 'e',   'オ': 'o',
        'ァ': 'la',  'ィ': 'li',  'ゥ': 'lu',  'ェ': 'le',  'ォ': 'lo', 
        'カ': 'ka',  'キ': 'ki',  'ク': 'ku',  'ケ': 'ke',  'コ': 'ko',
        'ガ': 'ga',  'ギ': 'gi',  'グ': 'gu',  'ゲ': 'ge',  'ゴ': 'go',
        'サ': 'sa',  'シ': 'si',  'ス': 'su',  'セ': 'se',  'ソ': 'so',
        'ザ': 'za',  'ジ': 'ji',  'ズ': 'zu',  'ゼ': 'ze',  'ゾ': 'zo',
        'タ': 'ta',  'チ': 'ti',  'ツ': 'tu',  'テ': 'te',  'ト': 'to',
        'ダ': 'da',  'ヂ': 'di',  'ヅ': 'du',  'デ': 'de',  'ド': 'do',
        'ナ': 'na',  'ニ': 'ni',  'ヌ': 'nu',  'ネ': 'ne',  'ノ': 'no',
        'ハ': 'ha',  'ヒ': 'hi',  'フ': 'fu',  'ヘ': 'he',  'ホ': 'ho',
        'バ': 'ba',  'ビ': 'bi',  'ブ': 'bu',  'ベ': 'be',  'ボ': 'bo',
        'パ': 'pa',  'ピ': 'pi',  'プ': 'pu',  'ペ': 'pe',  'ポ': 'po',
        'マ': 'ma',  'ミ': 'mi',  'ム': 'mu',  'メ': 'me',  'モ': 'mo',
        'ヤ': 'ya',               'ユ': 'yu',               'ヨ': 'yo',
        'ャ': 'lya',              'ュ': 'lyu',              'ョ': 'lyo',
        'ラ': 'ra',  'リ': 'ri',  'ル': 'ru',  'レ': 're',  'ロ': 'ro',
        'ワ': 'wa',  'ヰ': 'i'  ,              'ヱ': 'e',   'ヲ': 'wo',
        'ヴ': 'bu',  'ッ': 'Q',
        'ン': 'nn', 
    }
    _KANA2CHAR2 = {
        'キャ': 'kya',                 'キュ': 'kyu',                   'キョ': 'kyo',
        'ギャ': 'gya',                 'ギュ': 'gyu',                   'ギョ': 'gyo',
        'クゥ': 'ku',
        'シャ': 'sha',                 'シュ': 'shu',                   'ショ': 'sho',
        'ジャ': 'ja',                  'ジュ': 'ju',                    'ジョ': 'jo',
        'チャ': 'cha',                 'チュ': 'chu',  'チェ': 'che',   'チョ': 'cho',
        'ティ': 'thi',                 'トゥ': 'tolu',
        'ディ': 'di',                  'デュ': 'dyu',  'ドゥ': 'd u',
        'ニャ': 'nya',                 'ニュ': 'nyu',                   'ニョ': 'nyo',
        'ファ': 'fa',   'フィ': 'fi',                  'フェ': 'fe',    'フォ': 'fo',
                                      'フュ': 'hyu',                   'フョ': 'hyo',
        'ヒャ': 'hya',                 'ヒュ': 'hyu',                   'ヒョ': 'hyo',
        'ビャ': 'bya',                 'ビュ': 'byu',                   'ビョ': 'byo',
        'ピャ': 'pya',                 'ピュ': 'pyu',                   'ピョ': 'pyo',
        'ミャ': 'mya',                 'ミュ': 'myu',                   'ミョ': 'myo',
        'リャ': 'rya',                 'リュ': 'ryu',                   'リョ': 'ryo',
        'ウァ': 'wha',  'ウィ': 'wi',                   'ウェ': 'we',   'ウォ': 'who',
        'ヴァ': 'va',   'ヴィ': 'vi',                   'ヴェ': 've',   'ヴォ': 'vo',
        'ウ゛ァ': 'va', 'ウ゛ィ': 'vi',                 'ウ゛ェ': 've', 'ウ゛ォ': 'vo'
    }
    _SYMBOL = {
        '.': ' ',
        'ー': ' ',
        '―': ' ',
        '‐': ' ',
        '—': ' ',
        '–': ' ',
        '−': ' ',
        '-': ' ',
        '_': ' ',
        '+': 'plus',
        '#': ' ',
        '^': ' ',
        '～': ' ',
        '〜': ' ',
        '~': ' ',
        '?': ' ',
        '!': ' ',
        '・': ' ',
        '”': ' ',
        '"': ' ',
        ':': ' ',
        ';': ' ',
        '$': ' ',
        '£': ' ',
        '€': ' ',
        '¢': ' ',
        '/': ' ',
        '|': ' ',
        '=': ' ',
        'ç': 'c',
        'с': 'c',
        'ć': 'c',
        'Ç': 'C',
        'é': 'e',
        'è': 'e',
        'ê': 'e',
        'ë': 'e',
        'ё': 'e',
        'ệ': 'e',
        'ế': 'e',
        'É': 'E',
        'È': 'E',
        'Ē': 'E',
        'Ë': 'E',
        'Ê': 'E',
        'ü': 'u',
        'ú': 'u',
        'û': 'u',
        'ù': 'u',
        'Ü': 'U',
        'Ú': 'U',
        'Û': 'U',
        'Ù': 'U',
        'á': 'a',
        'à': 'a',
        'ä': 'a',
        'ã': 'a',
        'â': 'a',
        'å': 'a',
        'À': 'A',
        'Á': 'A',
        'Å': 'A',
        'Â': 'A',
        'Ã': 'A',
        'κ': 'k',
        'к': 'k',
        'ñ': 'n',
        'ñ': 'n',
        'Ñ': 'N',
        'õ': 'o',
        'ó': 'o',
        'ò': 'o',
        'ö': 'o',
        'ő': 'o',
        'ô': 'o',
        'Ö': 'O',
        'Ó': 'O',
        'Õ': 'O',
        'Ò': 'O',
        'Р': 'P',
        'ş': 's',
        'š': 's',
        'Š': 'S',
        'ı': 'i',
        'ì': 'i',
        'í': 'i',
        'î': 'i',
        'ï': 'i',
        'Í': 'I',
        'Î': 'I',
        'Ì': 'I',
        'Ï': 'I',
        'Ý': 'Y',
        'у': 'y',
        'ž': 'z',
        'Ž': 'Z',

        # Greek letters
        'Α': 'alpha',
        'α': 'alpha',
        'Β': 'beta',
        'β': 'beta',
        'Γ': 'gamma',
        'γ': 'gamma',
        'Δ': 'delta',
        'δ': 'delta',
        'Ε': 'epsilon',
        'ε': 'epsilon',
        'Ζ': 'zeta',
        'ζ': 'zeta',
        'Η': 'eta',
        'η': 'eta',
        'Θ': 'theta',
        'θ': 'theta',
        'Ι': 'iota',
        'ι': 'iota',
        'Κ': 'kappa',
        'κ': 'kappa',
        'Λ': 'lambda',
        'λ': 'lambda',
        'Μ': 'mu',
        'µ': 'mu',
        'Ν': 'nu',
        'ν': 'nu',
        'Ξ': 'xi',
        'ξ': 'xi',
        'Ο': 'omicron',
        'ο': 'omicron',
        '∏': 'pi',
        'π': 'pi',
        'Ρ': 'rho',
        'ρ': 'rho',
        '∑': 'sigma',
        'Σ': 'sigma',
        'σ': 'sigma',
        'ς': 'sigma',
        'Τ': 'tau',
        'τ': 'tau',
        'Υ': 'upsilon',
        'υ': 'upsilon',
        'Φ': 'phi',
        'ϕ': 'phi',
        'φ': 'phi',
        'Χ': 'chi',
        'χ': 'chi',
        'Ψ': 'psi',
        'ψ': 'psi',
        'Ω': 'omega',
        'ω': 'omega',


        '’': ' ',
        '@': ' ',
        '©': ' ',
        '*': ' ',
        '◦': ' ',
        '°': ' ',
        '·': ' ',
        '•': ' ',
        '▪': ' ',
        '©': ' ',
        'º': ' ',
        '«': ' ',
        '…': ' ',
        '\\':' ',
        '⇒': ' ',
        '│': ' ',
        'Є': ' ', 
        '&': 'and',
        ',': ' ',
        '%': ' percent',
        '{': ' ', '}': ' ',
        '(': ' ', ')': ' ',
        '<': ' ', '>': ' ', 
        '[': ' ', ']': ' ',
        '。': ' ', '、': ' ', '，': ' ',
        '「': ' ', '」': ' ',
        '｛': ' ', '｝':' ',
        '＜': ' ', '＞': ' ',
        '（': ' ', '）': ' ',
        '【': ' ', '】': ' ',
        '『': ' ', '』': ' ',
        '\xad': '',
        # '=': 'ha'
    }

    _RE_HIRAGANA = re.compile(r'[ぁ-ゔ]')

    @classmethod
    def _preprocess(cls, ifile_name):
        digit_pattern = re.compile(r'\d')      
        output = ""
        n_lines = tools.count_lines(ifile_name)
        bar = progressbar.ProgressBar()

        with io.open(ifile_name, encoding='utf-8', errors='ignore') as f:
            for line in bar(f, max_value=n_lines):
                line = digit_pattern.sub('0', line)

                # convert from Zenkaku number and alphabet to Hankaku
                line = line.translate(cls._NUMBER_ZEN2HAN_TABLE)
                line = line.translate(cls._ALPHABET_ZEN2HAN_TABLE)

                for k in cls._SYMBOL.keys():
                    line = line.replace(k, cls._SYMBOL[k])
                output += line

        output = single_quote(output)
        output = output.replace('\n', ' .\n')
        output = re.sub(r' {2,}', r' ', output)

        return output

    @classmethod
    def single_quote(cls, text):
        sinqt_pattern = re.compile(r'([a-z])(\')([a-z])')
        quote = re.compile(r'\'')
        while True:
            match_obj = re.search(sinqt_pattern, text)
            if match_obj:
                new = match_obj.group(0).replace('\'', '7')
                text = text.replace(match_obj.group(0), new)
            else:
                break
        text = quote.sub('', text)

        return text        


    @classmethod
    def kana2alphab(cls, text):
        # convert from Hiragana to Katakana
        str_kana = cls._RE_HIRAGANA.sub(lambda x: chr(ord(x.group(0)) + 0x60), text)

        # convert from kana to alphabet
        for k in cls._KANA2CHAR2.keys():
            str_kana = str_kana.replace(k, cls._KANA2CHAR2[k])
        for k in cls._KANA2CHAR1.keys():
            str_kana = str_kana.replace(k, cls._KANA2CHAR1[k])

        # sokuon
        str_kana = re.sub(r'Q([a-z])', r'\1\1', str_kana)
        str_kana = re.sub(r'Q$', r'ltu', str_kana)

        return str_kana

    @classmethod
    def alphab2num(cls, text):
        # exchange upper case to lower case
        text = text.lower()

        # convert alphabet to number
        numseq = text.translate(cls._KEYIN2NUMBER_TABLE)
        return numseq

    @classmethod
    def delete_invalid_char(cls, line, output, rem_flag):
        reobject = re.compile(r'[0-9 .\n]*')

        if reobject.fullmatch(line) == None:
            rem_flag = True
            return output, rem_flag

        output.append(line)
        return output, rem_flag

    @classmethod
    def _English_corpus(cls, ifile_name):
        output = ""
        removed_Eng = ""
        eng_list = []
        out_list = []
        rem_cnt = 0

        n_lines = tools.count_lines(ifile_name)
        bar = progressbar.ProgressBar()

        with io.open(ifile_name, encoding='utf-8', errors='ignore') as f:
            for line in bar(f, max_value=n_lines):
                rem_flag = False
                line = line.lower()
                out_list, rem_flag = KeyinputFilter.delete_invalid_char(
                    KeyinputFilter.alphab2num(line), out_list, rem_flag)

                if rem_flag == True:
                    rem_cnt += 1
                else:
                    eng_list.append(line)
        removed_Eng = ''.join(eng_list)
        output = ''.join(out_list)

        # number of sequence that includes invalid letters
        print(">> invalid sequences: {num_rem}/{all}".format(num_rem=rem_cnt, all=n_lines))
        print("   ({} percent sentences are invalid.)".format(rem_cnt*100/n_lines))

        # write result
        _name = ifile_name.split(".")
        outpath = "{name}_removed.{ext}".format(name=_name[0], ext=_name[1])
        # removed English sequence
        with open(outpath, mode="w", encoding="utf-8") as f:
            f.write(removed_Eng)
            f.close()

        return output

    @classmethod
    def Japanese_corpus(cls, text):
        output = ""

        output = KeyinputFilter.kana2alphab(output)
        output = KeyinputFilter.alphab2num(output)
        output = KeyinputFilter.delete_invalid_char(output)
        output = KeyinputFilter.process_period(output)

        return output

    @classmethod
    def textfile_io(cls, ifile_name, ofile_name):
        """
        main

        input and output
        call some functions that have process you want to do
        """

        ######################################################
        result = KeyinputFilter._preprocess(ifile_name)
        # result = KeyinputFilter._English_corpus(ifile_name)
        # result = KeyinputFilter.Japanese_corpus(text_data)
        ######################################################

        # number sequence
        with open(ofile_name, "w", encoding="utf-8") as f:
            f.write(result)
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Character filter',
        usage = 'python tenkey_filter_V2.py -i [filename] -o [filename]',
        description = '',
        epilog = 'end',
        add_help = True,
        )

    parser.add_argument("-i", '--ifilename', help='file name of input text file')
    parser.add_argument("-o", "--ofilename", help='file name of output text file')
    args = parser.parse_args()
    KeyinputFilter.textfile_io(args.ifilename, args.ofilename)


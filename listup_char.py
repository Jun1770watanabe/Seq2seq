import argparse
from tqdm import tqdm

def check_include(in_char, char_list):
    for i in char_list:
        if i == in_char:
            return char_list
        else:
            pass
    char_list.append(in_char)
    return char_list

def main_process(ifile_pass, ofile_pass):
    """
    input: ifile_pass, ofile_pass (str)
        pass of input and output file

    process: This function list up all kind of characters
             in input file, and write the characters to output file.
             The list has no overlapping characters. 

             This program is for debugging, thus without file writing.
    """

    with open(ifile_pass, encoding="utf-8") as f:
        char_list = []
        for line in tqdm(f):
            text_list = [c for c in line]
            for i in text_list:
                char_list = check_include(i, char_list)
        f.close()

    with open(ofile_pass, 'w', encoding="utf-8") as f:
        for i in char_list:
            f.write(i)

def main2(ifile_pass, ofile_pass):
    with open(ifile_pass, encoding="utf-8") as f:
        output = ""
        cnt = 0
        for line in tqdm(f):
            if cnt == 100000:
                break
            output += line
            cnt += 1
        f.close()
    with open(ofile_pass, "w", encoding="utf-8") as f:
        f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'List up characters',
        usage = 'python listup_char.py -i [filename] -o [filename]',
        description = '',
        epilog = 'end',
        add_help = True,
        )

    parser.add_argument("-i", '--ifilename', help='file name of input text file')
    parser.add_argument("-o", "--ofilename", help='file name of output text file')
    args = parser.parse_args()
    main_process(args.ifilename, args.ofilename)
    # main2(args.ifilename, args.ofilename)

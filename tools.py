import io
import argparse



def count_lines(path):
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        return sum([1 for _ in f])

def make_dictionary(text):
    output = {}
    text_list = text.split("\n")
    for i in range(len(text_list)):
        output[i+1] = text_list[-i]
    return output

def insert_char(text):
        text.replace(".\n", " .\n") 

def write_row(text, num):
    output = ""

    text_dict = make_dictionary(text)
    for k in text_dict.keys():
        if k == num+1:
            break
        else:
            output += text_dict[k]
            output += "\n"
    return output

def textfile_io(ifile_name, ofile_name):
    """
    main

    input and output
    call some functions that have process you want to do
    """
    # read text data
    with open(ifile_name, encoding="utf-8") as f:
        text_data = f.read()
        f.close()

    result = ""
    ######################################################
    # result = tools_filewriting.insert_char(text_data)
    result = write_row(text_data, 10)
    ######################################################

    # write result
    with open(ofile_name, "w", encoding="utf-8") as f:
        f.write(result)
        f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Character filter',
        usage = 'python tools.py [filename]',
        description = '',
        epilog = 'end',
        add_help = True,
        )

    parser.add_argument("-i", '--ifilename', help='file name of input text file')
    parser.add_argument("-o", "--ofilename", help='file name of output text file')
    args = parser.parse_args()
    textfile_io(args.ifilename, args.ofilename)


import io
import ast 
import matplotlib.pyplot as plt

def plot_one_data(x, y):
    plt.plot(x, y, label="eng2fre", marker="o", linewidth=3)
    plt.legend(loc="lower right", fontsize=20)

    plt.title("English to French task", fontsize=15)
    plt.xlabel("iteration", fontsize=15)
    plt.ylabel("BLEU score", fontsize=15)
    plt.grid(which="both")
    
    plt.show()

def plot_two_data(x1, y1, x2, y2):
    plt.plot(x1, y1, label="model_5mil", marker="o", linewidth=3)
    plt.plot(x2, y2, label="model_20mil", marker="o", linewidth=3)
    plt.legend(loc="lower right", fontsize=20)

    plt.title("compare training data size", fontsize=15)
    plt.xlabel("iteration", fontsize=15)
    plt.ylabel("BLEU score", fontsize=15)
    plt.grid(which="both")
    
    plt.show()

def plot_three_data(x1, y1, x2, y2, x3, y3):
    plt.plot(x1, y1, label="model_mbs16", marker="o", linewidth=3)
    plt.plot(x2, y2, label="model_mbs32", marker="o", linewidth=3)
    plt.plot(x3, y3, label="model_mbs64", marker="o", linewidth=3)
    plt.legend(loc="lower right", fontsize=20)

    plt.title("compare mini-batch size", fontsize=15)
    plt.xlabel("iteration", fontsize=15)
    plt.ylabel("BLEU score", fontsize=15)
    plt.grid(which="both")
    
    plt.show()

def plot_four_data(x1, y1, x2, y2, x3, y3, x4, y4):
    plt.plot(x1, y1, label="model_mbs16", marker="o", linewidth=3)
    plt.plot(x2, y2, label="model_mbs32", marker="o", linewidth=3)
    plt.plot(x3, y3, label="model_mbs64", marker="o", linewidth=3)
    plt.plot(x4, y4, label="model_mbs128", marker="o", linewidth=3)
    plt.legend(loc="lower right", fontsize=20)

    plt.title("compare mini-batch size", fontsize=15)
    plt.xlabel("iteration", fontsize=15)
    plt.ylabel("BLEU score", fontsize=15)
    plt.grid(which="both")
    
    plt.show()

def read_dict_pass(path):
    i_key = "iteration"
    b_key = "validation/main/bleu"
    x = []
    y = []

    with io.open(path, encoding="utf-8") as f:
        text = f.read()
        log_dict = ast.literal_eval(text)
        for log in log_dict:
            if b_key in log:
                x.append(log[i_key])
                y.append(log[b_key]*100)

    return x,y


def main():
    # # one parameter
    # path = "model_0.1mil_original/log"

    # x,y = read_dict_pass(path)
    # plot_one_data(x,y)


    # two parameter
    path1 = "result/model_5mil_mbs128_v5/log"
    path2 = "result/model_20mil_mbs128_v5/log"

    x1,y1 = read_dict_pass(path1)
    x2,y2 = read_dict_pass(path2)
    plot_two_data(x1,y1,x2,y2)


    # # three parameter
    # path1 = "model_0.1mil_mbs32_v4/log"
    # path2 = "model_0.1mil_mbs64_v4/log"
    # path3 = "model_0.1mil_mbs128_v4/log"

    # x1,y1 = read_dict_pass(path1)
    # x2,y2 = read_dict_pass(path2)
    # x3,y3 = read_dict_pass(path3)
    # plot_three_data(x1,y1,x2,y2,x3,y3)


    # # four parameter
    # path1 = "model_0.1mil_mbs16/log"
    # path2 = "model_0.1mil_mbs32/log"
    # path3 = "model_0.1mil_mbs64/log"
    # path4 = "model_0.1mil_mbs128/log"

    # x1,y1 = read_dict_pass(path1)
    # x2,y2 = read_dict_pass(path2)
    # x3,y3 = read_dict_pass(path3)
    # x4,y4 = read_dict_pass(path4)
    # plot_four_data(x1,y1,x2,y2,x3,y3,x4,y4)

if __name__ == "__main__":
    main()
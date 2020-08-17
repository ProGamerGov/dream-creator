from matplotlib import pyplot
import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_file", type=str, default='train_acc.txt')
    parser.add_argument("-graph_title", type=str, default='Accuracy Values')
    parser.add_argument("-x_name", type=str, default='Epoch')
    parser.add_argument("-y_name", type=str, default='Accuracy')
    parser.add_argument("-class_strings", type=str, default='')
    params = parser.parse_args()
    main_func(params)


def main_func(params):
    with open(params.csv_file, 'r') as f:
        csv_data = csv.reader(f)
        data_values = list(zip(*[map(float, row) for row in csv_data]))
    data_values[0] = [int(x) for x in data_values[0]]

    NUM_COLORS = len(data_values)-1
    cm = pyplot.get_cmap('gist_rainbow')
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    pyplot.grid()
    pyplot.title(params.graph_title)
    pyplot.xlabel(params.x_name)
    pyplot.ylabel(params.y_name)

    class_strings = params.class_strings.split(',')
    marker = 'o'
    for x in range(len(data_values)-1):
        label = 'cat' + str(x) if params.class_strings == '' else class_strings[x]
        marker = 'o' if marker == ',' else ','
        pyplot.plot(data_values[0], data_values[x+1], label=label, lw=1, marker=marker)
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    pyplot.show()



if __name__ == "__main__":
    main()
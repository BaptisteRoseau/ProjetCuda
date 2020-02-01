import matplotlib.pyplot as plt
import sys
import csv

# DATA FORMAT:
#       TITLE, XTITLE, YTITLE, YSCALE ("linear" or "log")
#       BAR LABEL, value
#       BAR LABEL, value
#       ...


if len(sys.argv) < 2:
    print("Usage: python3 drawCurve.py <my_file1.csv> <my_file2.csv>..")
    exit(0)


for fileIdx in range(1, len(sys.argv)):
    data = []
    plt.figure(fileIdx)
    with open(sys.argv[fileIdx],'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  
        for row in reader:
            data.append(row)

    # Getting curve information
    title  = data[0][0]
    xtitle = data[0][1]
    ytitle = data[0][2]
    yscale = data[0][3]

    # Retrieving curves information
    labels = []
    i = 1
    while i < len(data):
        if len(data[i]) > 1:
            barTitle = str(data[i][0])
            barValue = float(data[i][1])
            plt.bar([i], [barValue])
            labels.append(barTitle)
        i += 1

    # Ploting legend
    plt.xticks(range(1, len(labels)+1), labels)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.yscale(str(yscale).replace(" ", ""))
    #plt.legend()
    plt.grid()
plt.show()
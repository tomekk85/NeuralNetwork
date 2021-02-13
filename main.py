import numpy as np
from neuralnet import *
import pandas as pd
from datapreparation import *
import random

np.random.seed(2)


# odczytanie danych z pliku zewnętrznego i zapisanie ich do obiektu dataframe
dframe = pd.read_csv('new_file1.csv', sep=';')
#dframe['cons.conf.idx'] = dframe['cons.conf.idx'].abs()
#dframe['emp.var.rate'] = dframe['emp.var.rate'].abs()
#dframe.loc[(dframe['pdays'] == -1, 'pdays')] = 22

#dframe.to_csv('new_file1.csv', sep=';')
print("Dane wejściowe")
print(dframe)
print(50*"-")
print("Odchylenie standardowe")
print(dframe.std())
print(50*"-")
print("Średnia arytmetyczna")
print(dframe.mean())
print(50*"-")
# standaryzacja z-score
# dframe = (dframe - dframe.mean()) / dframe.std()

for column in dframe:
    if column != 'y':
        dframe[column] = (dframe[column] - dframe[column].mean()) / dframe[column].std()
print("")
print("Dane po standaryzacji metodą z-score")
print(dframe)
print(50*"-")

columnNames = dframe.columns
# funkcja pomocnicza drukująca dane wejściowe z nazwami kolumn
def printInput(_input):
    global columnNames
    string =""
    for i in range(0, len(columnNames)):
        if(i % 5 == 0 and i != 0):
            string += "\n"
        string += columnNames[i] + "=" + str(_input[i]) + ", "
    print(string)


# utworzenie macierzy z danymi trenującymi z obiektu dataframe
train_arr = dframe.to_numpy(dtype=np.float64)



_input = train_arr[0]
lay_list = [len(_input) - 1, 17, 1]

net = Network(input_vector=_input, list_of_layers=lay_list, step_back_prop=0.2)
net.forward()
net.backward()
net.calculate_final_output()
weights = net.output_weights


for i in range(1, 10_000):
    rand_index = random.randint(0, 902)
    _input = train_arr[rand_index]
    net = Network(input_vector=_input, list_of_layers=lay_list, step_back_prop=0.2, input_weights=weights)
    net.teach()
    weights = net.output_weights
    if i % 250 == 0:
        net_2 = Network(input_vector=_input, list_of_layers=lay_list, step_back_prop=0.2, input_weights=weights)
        net_2.forward()
        print("iteracja: " + str(i))
        print("index=" + str(rand_index))
        #print("iteracja: " + str(i) + " input: " + str(_input) + " " + str(net_2.initial_output))
        printInput(_input)
        print("Odpowiedź sieci: " +str(net_2.initial_output))
        print(50*"-")

        '''
        for i in range(0, len(weights)):
            diff = net_2.output_weights[i] - weights[i]
            print(diff)
        '''

print(50*"-")
print("Po wytrenowaniu")
print(50*"-")
print(50*"-")

do_sprawdzenia = [train_arr[0], train_arr[450], train_arr[734], train_arr[800]]


for i in range(0, len(do_sprawdzenia)):
    _input = do_sprawdzenia[i]
    #_input = tr[401]
    print("Wektor wejściowy:")
    printInput(_input)
    net = Network(input_vector=_input, list_of_layers=lay_list, step_back_prop=0.2, input_weights=weights)
    net.forward()
    print("Odpowiedź sieci:")
    print(net.initial_output)


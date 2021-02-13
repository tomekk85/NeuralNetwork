import numpy as np
from neuralnet import *
import pandas as pd
from datapreparation import *

np.random.seed(0)


dp = DataPreparation('bank-additional.csv')
dp.prepare_data()
dframe = dp.df
print(dframe)
#print(dframe.columns)


def teach_net(network, iterations, train_matrix, in_weights):
    weights = in_weights
    #network.check()
    #print(network.output_weights)

    for i in range(1, iterations):
        input_ = train_matrix[i]

        network = Network(input_, list_of_layers, 0.2, weights)
        network.teach()
        weights = network.output_weights
        if i % 500 == 0:
            print('-' * 50)
            print("After " + str(i) + " iterations.")
            network.check()
            print('-' * 50)
    return weights

# odczytanie danych z pliku zewnętrznego i zapisanie ich do obiektu dataframe
#dframe = pd.read_csv('new_file.csv', sep=';')
#print(dframe.columns)
#dframe.to_csv('new_file.csv', sep=';')

# utworzenie macierzy z danymi trenującymi z obiektu dataframe

input_arr = dframe.to_numpy(dtype=np.float64)
#print(input_arr)



#uczenie sieci
# pierwszsy wektor z macierzy ternującej "podany" do trenowania



y = [2, 3, 4, 1, 8, 1]

input_ = input_arr[0]

list_of_layers = [len(input_) - 1, 90, 50, 20, 1]
net = Network(input_, list_of_layers, -1)
'''w_0 = net.input_weights
print("\ninput weights")
print(net.input_weights)
net.teach()
net.check()

print('\noutput weights')
print(net.output_weights)'''
net.teach()

#wagi po 4000 iteracji
w_1 = teach_net(network=net, iterations=4000, train_matrix=input_arr, in_weights=net.output_weights)

#print('\noutput weights-4000')
#print(w_1)

#x = [0.75, 0.6, 0.1, 0.1, 0., 0., 0., 0.1, 0.4, 0.4, 0.0109, 0.1, 0., 0.1, 0.1, -0.18,
 #    0.93075, -0.471, 0.1405, 0.50991, 0.]

#x = input_arr[128]
x = input_arr[128]

print("Po wytrenowaniu sieci wprowadzamy  wektor:")
print(x)

'''x = [0.3,   0.6,    0.1,    0.7,    0.,     0.1,    0.,     0.1,    0.7,    0.2,
 0.0229, 0.1,    0.,    0.2,    0.1,    1.]'''

network = Network(input_vector=x, list_of_layers=list_of_layers, step_back_prop=0.2, input_weights=w_1)
network.forward()
print("Odpowiedź wyternowanej sieci:")
print("y=" + str(network.initial_output))

#print diagnostics
'''
net.forward()
net.initial_output = net.neurons[len(net.neurons) - 1]
print("weigths")
print(net.input_weights)

print("neur")
print(net.neurons)

net.backward()
net.calculate_final_output()
print("errors")
print(net.errors)
print("neurons - back")
print(net.neurons)


w_0 = net.input_weights.copy()
w_1 = net.output_weights.copy()

print('\nout_w - inp_w')
for i in range(0, len(w_1)):
    print('layer: ' + str(i))
    print(w_1[i] - w_0[i])
'''
'''
w_0 = net.input_weights.copy()
w_1 = net.output_weights.copy()

for i in range(0, len(w_1)):
    print('layer: ' + str(i))
    print(w_1[i] - w_0[i])
'''


'''
print('w0')
print(w_0)
print('w1')
print(w_1)
'''




'''
print("weights")
print(w)
print("errors")
print(network.errors)
print("neurons")
print(network.neurons)


print("input vect")
print(network.input_vector)
'''
'''
w = network.input_weights
weights = network.output_weights

print("input w")
print(w)
print("output w")
print(weights)
'''
#for i in range(0, len(w)):
    #print(w[i] - weights[i])


#przykład użytkownika
'''
print(network.output_weights)
network.check()

x = [0.3,  0.2,  0.1,  0.3,  0.0,  0.1,  0.0,  0.1,  
        0.5,  0.5,  0.0487,  0.2,  -0.1,  0.0,  0.0,  -0.18,  
        0.92893,  -0.462,  0.1313,  0.5099100000000001,  0.0]


network = Network(x, list_of_layers, 0.2, weights)
network.calculate_final_output()
print(network.final_output)
'''
'''
print("Po wytrenowaniu dla wektora wag:")
print(x)
print(network.get_final_output())
'''
'''
#zapisywanie wag do plików

df_weights_1 = pd.DataFrame(weights[0])
df_weights_2 = pd.DataFrame(weights[1])
df_weights_3 = pd.DataFrame(weights[2])



df_weights_total = [df_weights_1, df_weights_2, df_weights_3]

for i in range(0, len(df_weights_total)) :
    df_weights_total[i].to_csv('w' + str(i) + '.txt', index=False,sep=';')

'''
#odczyt wag z plików
'''
list = []

for i in range(0, 3):
    list.append(
        np.array(
            pd.read_csv('w' + str(i) + '.txt', sep=';')
        )
    )

print(list)
'''
# df_weights.to_csv('weights.csv', sep=';')

# network = Network(przyklad, przyklad_weights, przyklad_layers, 0.2)
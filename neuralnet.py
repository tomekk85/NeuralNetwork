import numpy as np


def prop_function(value):
    """Metoda statyczna - funkcja aktywacji"""
    return round(
        1 / (1 + np.e ** (-value)), 5
    )


#
class MiddleLayer:
    """klasa pomocnicza - warstwa pośrednia oblicza wektor z wartościami neuronów na wyjściu warstwy"""
    def __init__(self, input_vector, layer_neurons_quantity, initial_weights, bias=0):
        """

        :param input_vector: wektor neuronów na "wejściu" warstwy
        :param layer_neurons_quantity:
        :param initial_weights:
        :param bias:
        pole output przechowuje wektor neuronów na "wyjściu" warstwy
        """
        self.input_vector = np.array(input_vector)
        self.layer_neurons_quantity = np.array(layer_neurons_quantity)
        self.initial_weights = np.array(initial_weights)
        self.bias = bias
        self.output = []

    def forward(self):
        """metoda wykonuje propagację w przód dla danej warstwy"""
        self.output = np.dot(self.input_vector, self.initial_weights)

        for i in range(0, len(self.output)):
            self.output[i] = self.output[i] - self.bias

        for i in range(0, len(self.output)):
            self.output[i] = prop_function(self.output[i])


#
class Network:
    """klasa właściwa - odzwierciedla sieć neuronową"""

    def __init__(self, input_vector, list_of_layers, step_back_prop, input_weights=[]):
        """

        :param input_vector: wektor wejściowy - tablica 1-wymiarowa,
                                gdzie na ostatniej pozycji jest oczekiwana wartość na "wyjściu"
        :param list_of_layers: tablica przechowująca informację o liczbie warstw(długość tablicy)
                                oraz liczbie neuronów na każdej warstwie
        :param step_back_prop: parametr uczenia -(p)
        :param input_weights: wagi na wejściu( jeżeli nie podamy tego parametru to przekazana zostanie pusta tablica,
                                co poskutkuje wygenerowaniem losowych wag)
        :neurons - lista neuronów
        :errors - lista błędów dla neuronów
        :output_weights - wagi końcowe - po propagacji wstecznej
        :initial_output: wartość na "wyjściu" sieci po propagacji w przód dla "starych" wag
        :final_output: wartość na "wyjściu" sieci po propagacji w przód dla "nowych" wag
        """
        self.input_vector = input_vector[0: len(input_vector) - 1]
        self.expected_output = input_vector[len(input_vector) - 1]
        self.list_of_layers = list_of_layers
        self.step_back_prop = step_back_prop
        self.input_weights = []
        self.neurons = []
        #self.neurons.append(np.array(self.input_vector))
        self.errors = []
        self.output_weights = []
        self.initial_output = []
        self.final_output = []

        if len(input_weights) == 0:
            self.input_weights = self.generate_random_weights()
        else:
            self.input_weights = input_weights

        self.output_weights = self.generate_shape_of_output_weights()

    def teach(self):
        self.forward()
        self.backward()
        self.calculate_final_output()

    def forward(self):
        """Metoda przeprowadza propagację sieci w przód"""
        self.calculate_neurons(self.input_weights)
        self.initial_output = self.neurons[len(self.neurons) - 1]

    def backward(self):
        """Metoda przeprwoadza propagację sieci w tył"""
        self.calculate_errors()
        self.calculate_weights()

    def calculate_neurons(self, weights):
        """metoda kalkuluje listę neuronów całej sieci i zapisuje ją do pola klasy Network o nazwie neurons"""
        neur = []
        neur.append(np.array(self.input_vector))
        for i in range(1, len(self.list_of_layers)):
            middle = MiddleLayer(neur[i - 1], self.list_of_layers[i], weights[i - 1], 0.002)
            middle.forward()
            # self.neurons.append(np.array(middle.output))
            neur.append(np.array(middle.output))

        self.neurons = neur

    def calculate_errors(self):
        """metoda tworzy listę błędów, dla poszczególnych neuronów i zapisuję ją do pola klasy Network o nazwie errors"""
        self.errors = []

        # stworzenie wyzerowanej listy o odpowiednim "kształcie" (takim samym jak lista neuronów)
        for i in range(0, len(self.neurons)):
            self.errors.append(np.zeros((len(self.neurons[i]))))

        # początek obliczeń
        last_row = len(self.errors) - 1  # warstwa końcowa

        for i in range(0, len(self.errors[last_row])):
            # derrivative f'(Sn) = Un * (1 - Un)
            self.errors[last_row][i] = self.neurons[last_row][i] * (1 - self.neurons[last_row][i])
            # error = (C - Un) * f'(Sn)
            self.errors[last_row][i] = (self.expected_output - self.neurons[last_row][i]) * self.errors[last_row][i]

        #for i in range(len(self.errors) - 2, -1, -1):
        # pozostałe warstwy
        counter = 0
        for i in range(len(self.errors) - 2, -1, -1):
            for j in range(0, len(self.errors[i])):
                # derrivative f'(Sn) = Un * (1 - Un)
                self.errors[i][j] = self.neurons[i][j] * (1 - self.neurons[i][j])
                # error k = Waga(nr górnej warstwy, k) * błąd(numer górnej warstwy) * f'(Sn)
                s = self.errors[i][j]
                w = self.input_weights[i][j]
                sum = 0

                err = self.errors[i + 1]
                for k in range(0, len(err)):
                    sum += w[k] * err[k]
                    '''print(str(counter) + " i=" + str(i) + " j=" + str(j) + " k=" + str(k) + " sum=" + str(sum)
                          + "\ts=" + str(s) + "\tw[k]=" + str(w[k]) + "\terr[k]=" + str(err[k])
                          + "\tneur[i,j]=" + str( self.neurons[i][j]))'''

                self.errors[i][j] = sum * s
                #print(str(counter) +" " + str(i))

                counter +=1

    def calculate_weights(self):
        """Metoda tworzy listę nowych wag i zapisuje ją do pola klasy Network output_weights"""
        o_w = self.output_weights
        i_w = self.input_weights
        err = self.errors
        n = self.neurons
        p = self.step_back_prop

        for i in range(0, len(o_w)):
            for j in range(0, len(o_w[i])):
                for k in range(0, len(o_w[i][j])):
                    o_w[i][j][k] = i_w[i][j][k] + p * err[i + 1][k] * n[i][j]

    def check(self):
        """metoda wypisuje wartość na "wyjściu" ostatniej warstwy sieci przed zmianą wag
        oraz po zmianie wag, a także wypisuje różnicę"""
        input_neuron_last_pos = self.neurons[len(self.neurons) - 1]

        #self.calculate_neurons(self.output_weights)
        output_neuron_last_pos = self.neurons[len(self.neurons) - 1]
        '''
        print(self.initial_output)
        print(self.final_output)
        print(self.final_output - self.initial_output)
        '''

        print("initial network output: " + str(self.initial_output))
        print("final network output: " + str(self.final_output))
        print("difference: " + str(self.final_output - self.initial_output))

    def generate_random_weights(self):
        """metoda generuje losowe wagi"""
        weights = []
        for i in range(1, len(self.list_of_layers)):
            weights.append(
                np.array(
                    np.random.uniform(
                        low=-1.0, high=1.0, size=(self.list_of_layers[i - 1], self.list_of_layers[i])))
            )
        return weights

    def generate_shape_of_output_weights(self):
        """metoda generauje listę wag na wyjściu o odpowiednich rozmiarach wypełnioną zerami"""
        weights = []
        for i in range(1, len(self.list_of_layers)):
            weights.append(
                np.array(np.random.uniform(
                    low=0, high=0, size=(self.list_of_layers[i - 1], self.list_of_layers[i])))
            )
        return weights

    def calculate_final_output(self):
        """metoda generauje wartość na wyjściu sieci"""
        self.calculate_neurons(self.output_weights)
        self.final_output = self.neurons[len(self.neurons) - 1]

    def get_init_output(self):
        return self.initial_output

    def get_final_output(self):
        return self.final_output



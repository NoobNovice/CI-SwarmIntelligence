import numpy
import random
import math
import copy

class SwarmIntelligence:

    __agent_arr = []  # test
    __num_agent = 0
    __mlp_structure = []
    __report = None
    __search_time = 0

    def __init__(self, time, num_agent):
        self.__num_agent = num_agent
        self.__search_time = time
        self.__report = open("Report.txt", "a")
        return

    def init_mlp(self, arr_structure, rang_of_population):
        self.__mlp_structure = arr_structure
        w = arr_structure[0]
        for i in range(1, len(arr_structure)):
            w += arr_structure[i] + (arr_structure[i] * arr_structure[i - 1])
        # init population
        for i in range(self.__num_agent):
            arr = []
            for j in range(w):
                if j < arr_structure[0]:
                    arr.append(0)
                else:
                    arr.append(random.uniform(-rang_of_population, rang_of_population))
            self.__agent_arr.append(arr)
        return

    def star_train(self, arr_data, arr_design_output):  # number of arr_data must equal number of arr_design_output
        t = 0
        p_best = self.init_arr(9999, len(self.__agent_arr))
        performance_best = copy.deepcopy(self.__agent_arr)
        while t < self.__search_time:
            print(t)
            # calculate performance
            performance = []
            for agent in self.__agent_arr:
                error = 0
                for index in range(len(arr_data)):
                    output = self.feed_forward(arr_data[index], agent)
                    error += self.fitness_error(output, arr_design_output[index])
                performance.append(error / len(arr_data))

            # calculate person best
            for i in range(len(performance)):
                if performance[i] < p_best[i]:
                    p_best[i] = performance[i]
                    performance_best[i] = self.__agent_arr[i]

            # find global best
            g_best_index = performance.index(min(performance))

            # update point
            for i in range(len(self.__agent_arr)):
                # edit vector
                v_p = numpy.multiply(numpy.subtract(performance_best[i], self.__agent_arr[i]), 0.3)
                v_g = numpy.multiply(numpy.subtract(self.__agent_arr[g_best_index], self.__agent_arr[i]), 0.7)
                temp = numpy.add(v_p, v_g)
                # update vector
                self.__agent_arr[i] = numpy.add(self.__agent_arr[i], temp)
            t += 1
        return performance_best[p_best.index(min(p_best))]

    def feed_forward(self, arr_input, chromosome):
        # output of each layer
        input_of_layer = arr_input

        # point index
        weight_index = self.__mlp_structure[0]
        bias_index = self.__mlp_structure[0]

        for layer_index in range(1, len(self.__mlp_structure)):  # each layer
            arr = []
            bias_index += self.__mlp_structure[layer_index] * self.__mlp_structure[layer_index - 1]

            for node_index in range(self.__mlp_structure[layer_index]):  # each node
                v = 0
                for input_index in range(self.__mlp_structure[layer_index - 1]):  # each wire. number of wire must equal number of input
                    i_index = input_index
                    w_index = weight_index + (self.__mlp_structure[layer_index] * input_index)
                    v += chromosome[w_index] * input_of_layer[i_index]
                b_index = bias_index
                v += chromosome[b_index]
                arr.append(self.activation_func(v))
                # update index
                bias_index += 1
                weight_index += 1
            weight_index = bias_index
            input_of_layer = arr

        return input_of_layer

    # sigmoid function
    @staticmethod
    def activation_func(v):
        y = 1 / (1 + math.exp(-v))
        return y

    @staticmethod
    def fitness_error(output, design_output):
        error_output = numpy.subtract(design_output, output)
        e = numpy.sum(numpy.fabs(error_output)) / len(error_output)
        return e

    # this function use structure output node is 1
    def test_mean_square_error(self, input_set, design_output,  mlp_arr, round_test):
        result = 0
        self.__report.write("fold {}\n".format(round_test))
        self.__report.write("MLP weight: {}\n".format(mlp_arr))
        self.__report.write("design output, prediction output\n")
        for cur_row in range(0, len(input_set)):
            output = self.feed_forward(input_set[cur_row], mlp_arr)
            result += abs(design_output[cur_row][0] - output[0])  # one output node
            self.__report.write("{},{}\n".format(design_output[cur_row], output))
        result = result/len(input_set)
        self.__report.write("{}\n\n".format(result))
        return

    @staticmethod
    def init_arr(value, size):
        return [value] * size

from SwarmIntelligence import SwarmIntelligence
from ReadExcelFile import ReadExcelFile
import copy
import numpy
from test import ProcessData

data = ReadExcelFile("test_cut.xls", 12)
data.ten_fold_data()

for i in range(0, 10):
    # prepare data
    fold_data = copy.deepcopy(data.fold_data)
    test_data = fold_data.pop(i)
    train_data = numpy.concatenate(fold_data)
    # init module
    data_input, design_output = data.split_output(train_data)
    swarm = SwarmIntelligence(20, 1000)
    swarm.init_mlp([12, 6, 2, 1], 5)
    # test module
    module = swarm.star_train(data_input, design_output)
    print(module)
    test_input, test_output = data.split_output(test_data)
    swarm.test_mean_square_error(test_input, test_output, module, i + 1)

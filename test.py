import copy
import numpy
import xlwt
import math

class ProcessData:

    data = []

    def __init__(self, data_set):
        self.data = numpy.asarray(data_set)
        return

    def missing_mean_fill(self):
        self.data = self.data.transpose()
        for cur_row in range(len(self.data)):
            data_sum = 0
            count = 0
            missing_index = []
            # find mean
            for cur_col in range(len(self.data[0])):
                if self.data[cur_row][cur_col] != -200:
                    data_sum += self.data[cur_row][cur_col]
                    count += 1
                else:
                    missing_index.append(cur_col)
            mean = data_sum / count
            for index in missing_index:
                self.data[cur_row][index] = mean
        self.data = self.data.transpose()
        return

    def cut_missing(self):
        result = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j] == -200:
                    break
                if j == len(self.data[i]) - 1:
                    result.append(self.data[i])
        self.data = numpy.asarray(result)
        return

    def decimal_scale(self):
        self.data = self.data.transpose()
        for i in range(len(self.data)):
            max_num = max(self.data[i])
            p = 0
            while max_num/10 > 1:
                print(i)
                print(max_num)
                p += 1
                max_num = max_num / 10
            for j in range(len(self.data[i])):
                self.data[i][j] = self.data[i][j] / math.pow(10, p)
        self.data = self.data.transpose()
        return

    def export_excel(self, filename):
        book = xlwt.Workbook(encoding="utf-8")
        sh = book.add_sheet("sheet1")

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                sh.write(i, j, round(self.data[i][j], 2))

        book.save(filename)
        return

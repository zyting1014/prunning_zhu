'''
    用来记录剪枝中的统计量，转成csv
'''
import pandas as pd
import datetime


def save(current_ratio_list, whole_inference_time, sum_list, top1_err_list, parameter_ratio_list):
    #  current_ratio_list : current ratio列表
    #  whole_inference_time : 整个网络推理时间列表（比sum大一点）
    #  sum : 各层forward函数的累计列表
    #  top1_err_list : top1 error列表
    #  parameter_ratio_list ： parameter_ratio列表
    print("保存数据中...")

    my_dict = {"current_ratio_list": current_ratio_list,
               "whole_inference_time": whole_inference_time,
               "sum_list": sum_list,
               "top1_err_list": top1_err_list,
               "parameter_ratio_list": parameter_ratio_list}
    my_list = [datetime.datetime.now().year, datetime.datetime.now().month,
               datetime.datetime.now().day, datetime.datetime.now().hour,
               datetime.datetime.now().minute, datetime.datetime.now().second]
    filename = ""
    for item in my_list:
        filename = filename + str(item) + "_"
    filename = filename[:len(filename) - 1]
    filename = "../csv/" + filename + ".csv"
    data = pd.DataFrame(my_dict)

    data.to_csv(filename, encoding="utf_8_sig")
    print(filename + "已保存！")
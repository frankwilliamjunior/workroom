import numpy as np

def Bubble_sort(input_list):
    if len(input_list) == 0 or len(input_list) == 1:
        print("input list is None or just one element!")
        return input_list
    L = len(input_list)
    for i in range(L-1):
        for j in range(L-i-1):
            if input_list[j] > input_list[j+1]:
                temp = input_list[j]
                input_list[j] = input_list[j+1]
                input_list[j+1] = temp
    print(input_list)
    return input_list

def Quick_sort(input_list):
    if len(input_list) == 0 or len(input_list) == 1:
        print("input list is None or just one element!")
        return input_list
    L = len(input_list)
    pivot = input_lis[int(L/2)]
    

def Select_sort(input_list)
    if len(input_list) == 0 or len(input_list) == 1:
        print("input list is None or just one element!")
        return input_list


def Insert_sort(input_list):


Bubble_sort(test2)
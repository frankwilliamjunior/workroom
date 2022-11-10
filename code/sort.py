import numpy as np


# 快速排序
def quick_sort(array):
    def quicksort(array,start,end):

        if start >= end:
            return 
        # 基准值
        pivot = array[start]

        low = start
        high = end
        # 利用游标遍历
        while low < high:
            while low < high and array[high] > pivot:
                high -= 1

            array[low] = array[high]
            while low < high and array[low] < pivot:
                low += 1
            array[high] = array[low]

        array[low] = pivot

        quicksort(array,start,low-1)
        quicksort(array,low+1,end)
    quicksort(array,0,len(array)-1)
    return array

# 选择排序
def select_sort(array):
    for i in range(len(array)-1):
        index = i
        for j in range(i+1,len(array)):
            if array[j] < array[index]:
                index = j
        temp = array[index]
        array[index] = array[i]
        array[i] = temp
    return array

# 冒泡排序
def Bubble_sort(array):

    for i in range(len(array)-1):
        for j in range(len(array)-i-1):
            if array[j]> array[j+1]:
                temp = array[j]
                array[j] = array[j+1]
                array[j+1] = temp
    return array


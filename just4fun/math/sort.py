def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1  # 左子节点
    r = 2 * i + 2  # 右子节点

    # 找最大值的位置
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换
        heapify(arr, n, largest)  # 递归调整子树


def heap_sort(arr):
    n = len(arr)

    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 一个个取出堆顶元素（最大值）
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # 交换最大值到末尾
        heapify(arr, i, 0)  # 重新调整堆

    return arr


def shell_sort(arr):
    n = len(arr)
    gap = n // 2  # 初始步长为数组长度的一半

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            # 类似插入排序的过程
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
                print("-----")
            arr[j] = temp
        gap //= 2  # 缩小步长
    return arr


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)


if __name__ == '__main__':
    arr = [23, 12, 1, 8, 34, 54, 2, 3]
    print(shell_sort(arr))
    print(quicksort(arr))
    # 输出: [1, 2, 3, 8, 12, 23, 34, 54]
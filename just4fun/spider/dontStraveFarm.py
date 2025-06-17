with open('饥荒作物.txt', "r") as f:
    lines = f.readlines()

# print(lines)
fertilizer = ['催长剂', '堆肥', '粪肥']
season = ['春', '夏', '秋', '冬']


class Plant:
    def __init__(self, txt=''):
        self.name = txt[0: txt.index('(')]
        self.info = self.name + ' '
        seasonInfo = txt[txt.index('('): txt.index(')')]
        self.seasons = [seasonInfo.__contains__('春'), seasonInfo.__contains__('夏'),
                        seasonInfo.__contains__('秋'), seasonInfo.__contains__('冬')]
        for i in range(0, 4):
            if self.seasons[i] == 1:
                self.info += season[i] + ' '
        nums = txt[txt.index(')'):]
        self.nums = []
        symbol = 1
        for c in nums:
            if c == '-':
                symbol = -1
            if '0' <= c <= '9':
                self.nums.append(symbol * (ord(c) - ord('0')))
                symbol = 1
        for n in self.nums:
            self.info += str(n) + ' '

    def __str__(self):
        return self.info


plants = [Plant(l) for l in lines]
seasonAllows = {}
for p in plants:
    print(p)
    for i in range(0, 4):
        if p.seasons[i] == 1:
            if not seasonAllows.__contains__(season[i]):
                seasonAllows[season[i]] = [p]
            else:
                seasonAllows[season[i]].append(p)

for i in range(0, 4):
    print(season[i]+': ', end='')
    for p in seasonAllows[season[i]]:
        print(p.name + ' ', end='')
    print()

def printComb(selected):
    for i in range(3):
        print(selected[i].name + ' ', end='')
    print()

def selectAndCalculate(plantPot, perfects, lessPerfects, selected, n=3):
    if n > 0:
        if len(selected) == 0:
            rangeStart = 0
        else:
            rangeStart = plantPot.index(selected[-1])
        for i in range(rangeStart, len(plantPot) - n + 1):
            selected.append(plantPot[i])
            selectAndCalculate(plantPot, perfects, lessPerfects, selected, n-1)
            selected.pop()
        return
    count = [0, 0, 0]
    for i in range(3):
        for j in range(len(fertilizer)):
            count[j] += selected[i].nums[j]
    if count == [0, 0, 0]:
        printComb(selected)
        perfects.append(selected)
    else:
        minus = 0
        for i in range(3):
            minus += (count[i] < 0) * count[i]
        if minus >= -1:  # 竟然没有有一点瑕疵的组合
            print(count, ' ', end='')
            printComb(selected)
            lessPerfects.append(selected)


for i in range(0, 4):
    print(season[i] + ": ")
    plantPot = seasonAllows[season[i]]
    selected3 = []    # 一个地皮内最少3个同类植物，一共9个
    perfects = []
    lessPerfects = []
    selectAndCalculate(plantPot, perfects, lessPerfects, selected3)

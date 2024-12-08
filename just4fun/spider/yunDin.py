from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def downloadHtmlAndSave(url='', filename=''):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        # 等待页面内容加载完成
        page.wait_for_load_state("networkidle")
        page_content = page.content()
        soup = BeautifulSoup(page_content, 'html.parser')
        with open(filename, 'w') as f:
            f.write(soup.prettify())

        # 关闭浏览器
        browser.close()


class Hero:
    name = None
    jobIndexes = []
    jobs = []
    isChosen = 0
    price = 0
    index = 0

    def __init__(self, name='', title='', price=0, index=0):
        self.price = price
        self.name = name
        self.title = title
        self.jobIndexes = []
        self.jobs = []
        self.index = index

    def addJob(self, job=''):
        self.jobs.append(job)

    def printSelf(self):
        print(self.price, self.name, self.title, self.jobs)

    def initialJobIndexes(self, synergies):
        for j in self.jobs:
            index = 0
            for s in synergies:
                if j == s.name:
                    self.jobIndexes.append(index)
                    break
                index += 1
        self.jobIndexes.sort()


class Synergy:
    nums = []
    heroIndexes = []
    name = ''

    def __init__(self, name=''):
        self.name = name
        self.nums = []
        self.heroIndexes = []

    def addNum(self, n=0):
        self.nums.append(n)

    def printSelf(self):
        print(self.name, self.nums)

    def initialHeroIndexes(self):
        index = 0
        for h in heroes:
            for j in h.jobs:
                if j == self.name:
                    self.heroIndexes.append(index)
                    break
            index += 1
        self.heroIndexes.sort()


def extractHeros(html=''):
    soup = BeautifulSoup(html, 'html.parser')
    elements = soup.select(".champion-item-big")
    index = 0
    for e in elements:
        title, name = e.select(".name")[0].text.strip().split(' ')
        price = int(e.select(".price")[0].text.strip()[0:2])
        hero = Hero(name, title, price, index)
        jobs = e.select(".race-job-name")
        for j in jobs:
            hero.addJob(j.text.strip())
        heroes.append(hero)
        index += 1


def extractSynergies(html=''):
    soup = BeautifulSoup(html, 'html.parser')
    elements = soup.select(".synergy-item")
    for e in elements:
        synergy = Synergy(e.select(".synergy")[0].text.strip())
        nums = e.select(".txt-level")
        for n in nums:
            synergy.addNum(int(n.text.strip()))
        synergies.append(synergy)


def addSynergy(synergy_set, hero):
    for j in hero.jobIndexes:
        if j in synergy_set:
            synergy_set[j] += 1
        else:
            synergy_set[j] = 1


def checkSynergy(first, hero):
    for j in hero.jobIndexes:
        if j < first:
            return 1
    return 0


def delSynergy(synergy_set, hero):
    for j in hero.jobIndexes:
        synergy_set[j] -= 1
        if synergy_set[j] == 0:
            synergy_set.pop(j)


def isPerfect(synergy_set={}):
    b = []
    b.extend(synergy_set.keys())
    b.sort()
    for i in b:
        for n in synergies[i].nums:
            if synergy_set[i] < n:  # 默认羁绊数越来越大
                return -i  # 返回第一个不完美羁绊index
            if synergy_set[i] == n:
                break
    return 1


perfect_list = set()
def printIfPerfect(hero_list=None, synergy_set=None):
    # 字典类型，key:value, 只能按key遍历，synergy_set[key]取对应value
    if synergy_set is None or synergy_set == {}:
        synergy_set = {}
        for h in hero_list:
            addSynergy(synergy_set, heroes[h])
    if isPerfect(synergy_set) == 1:
        # text = str(hero_list) + "\n"
        text = ''
        coin = 0
        for h in hero_list:
            text += heroes[h].name + str(heroes[h].price) + ' '
            coin += heroes[h].price
        text += '--' + str(coin) + '\n'
        for i in synergy_set:
            text += synergies[i].name + str(synergy_set[i]) + ' '
        text += '\n\n'
        hero_list = hero_list.copy()
        hero_list.sort()
        hero_list = tuple(hero_list)
        if hero_list not in perfect_list:
            perfect_list.add(hero_list)
        # else:
        #     text = str(hero_list) + '\n' + text
            print(text, end='')
            with open("perfect_" + str(perfect_num) + '.txt', 'a') as f:
                f.write(text)


# 谁能想到，多年前我竟然能写出如此简洁的组合函数，虽然是用递归
def C(n, nList, i=0, l=0):
    if n<1:
        print(nList)
        return
    while i<8:
        nList[l] = i
        i += 1
        C(n-1,nList,i, l+1)  # 保持n + l = n
    return


def perfectComb(n=8):
    if n < 1:
        return
    # 穷举法 C(60, 8) = 60*59*58*...*53 / (1*2*3*...*8) == 22亿,对所有组合判断是否完美
    # [0,1,3,14,31,49,53,56]
    list_len = len(heroes)
    if list_len < n:
        return
    # 从0到listLen选择numToChose个数的所有可能选法，每个组合都执行func(该组合, kwargs)
    chosen_list = [-1] * n
    num = 0  # 已选择个数
    back = 0
    synergy_set = {}
    while chosen_list[0] <= list_len - n:  # 人数满n人就判断是否是最后一个组合
        if back == 1:  # 这种情况更多
            delSynergy(synergy_set, heroes[chosen_list[num - 1]])
            next_h = chosen_list[num - 1] + 1
            if next_h >= list_len - n + num:
                num -= 1
                if num == 0 and chosen_list[0] == list_len - n:
                    break
                continue
            chosen_list[num - 1] = next_h
            addSynergy(synergy_set, heroes[next_h])
            back = 0
        else:
            if num == 0:
                next_h = chosen_list[0] + 1
            else:
                next_h = chosen_list[num - 1] + 1
            if next_h > list_len - n + num:
                back = 1
                continue
            chosen_list[num] = next_h
            addSynergy(synergy_set, heroes[next_h])
            num += 1
        if num == n:
            printIfPerfect(chosen_list, synergy_set)
            back = 1


def removeSame(list1, l1, list2remove):
    for i in range(l1):
        if list1[i] in list2remove:
            list2remove.remove(list1[i])


def getLowest(hero_list, hl, s_index):
    # 次序颠倒的情况只会出现在分2次选择同一个羁绊中的英雄
    # 那就要每次选择时，保证次序不会颠倒，
    max = -1
    for i in range(hl):
        h = hero_list[i]
        for j in heroes[h].jobIndexes:
            if j == s_index and max < h:
                perfect = 1
                for j in heroes[h].jobIndexes:
                    if j > s_index:
                        perfect = 0
                        break
                if perfect == 1:
                    max = h
                break
    return max


def removeLowIndex(lowest, h_list):
    for i in range(len(h_list)):
        if h_list[i] == lowest:
            del h_list[:i + 1]
            return


def selectAddDo(h_list, num_to_chose, func, n, synergy_set, first, hero_list, hl):
    # hl = hero_list.len, first是第一个羁绊序号
    # 从h_list中选s_num个，加入hero_list中，更新synergy_set, 执行func(n, synergy_set, i, hero_list, hl)
    list_len = len(h_list)
    if list_len < num_to_chose:
        return
    num = 0  # 已选择个数
    back = 0
    chosen_list = [-1] * num_to_chose  # [8, 10, 18, 20, 45, 56, 59]
    while chosen_list[0] <= list_len - num_to_chose:  # 人数满n人就判断是否是最后一个组合
        if back == 1:  # 这种情况更多
            delSynergy(synergy_set, heroes[h_list[chosen_list[num - 1]]])
            next_h = chosen_list[num - 1] + 1
            if next_h >= list_len - num_to_chose + num:
                num -= 1
                if num == 0 and chosen_list[0] == list_len - num_to_chose:
                    break
                continue
            chosen_list[num - 1] = next_h
            hero_list[hl - 1 + num] = h_list[next_h]
            addSynergy(synergy_set, heroes[h_list[next_h]])
            if checkSynergy(first, heroes[h_list[next_h]]) == 1:
                continue
            back = 0
        else:  # 选择下一个
            if num == 0:
                next_h = chosen_list[0] + 1
            else:
                next_h = chosen_list[num - 1] + 1
            if next_h > list_len - num_to_chose + num:
                back = 1
                continue
            chosen_list[num] = next_h
            hero_list[hl + num] = h_list[next_h]
            addSynergy(synergy_set, heroes[h_list[next_h]])
            num += 1
            if checkSynergy(first, heroes[h_list[next_h]]) == 1:
                back = 1
                continue
        if num == num_to_chose:
            func(n, synergy_set, first, hero_list, hl + num_to_chose)
            back = 1


def perfectCombSmart(n=8, synergy_set=None, first=0, hero_list=None, hl=0):
    # 先选第一个羁绊，假设羁绊数（2，4，6，8）先从8个人里选择2个，根据不完美的羁绊继续选，直到人数超过n仍然不完美
    # first是第一个羁绊序号，后面加的英雄带有的羁绊不能低于first，否则会重复
    if hl == n:
        printIfPerfect(hero_list, synergy_set)
        return
    else:
        if synergy_set is None:
            synergy_set = {}
            hero_list = [-1] * n
    p = isPerfect(synergy_set)
    if p == 1:
        for i in range(first, len(synergies)):
            s = synergies[i]
            for ns in s.nums:
                lowest = -1
                if i in synergy_set:  # 增加的英雄序号不能低于已存在的同羁绊的英雄序号
                    if synergy_set[i] >= ns:
                        continue
                    s_num = ns - synergy_set[i]
                    lowest = getLowest(hero_list, hl, i)
                else:
                    s_num = ns
                if s_num + hl > n:
                    break
                h_list = s.heroIndexes.copy()
                # removeLowIndex(lowest, h_list)
                # 找到所有相同羁绊的英雄组，当多个英雄出现时，必须按顺序组合，否则跳过
                removeSame(hero_list, hl, h_list)
                selectAddDo(h_list, s_num, perfectCombSmart, n, synergy_set, i, hero_list, hl)
                break
    else:  # 优先补全羁绊
        p = -p  # p是不完美羁绊, 如果无法完美，则这条路走不下去
        if p < first:
            return
        for ns in synergies[p].nums:
            if synergy_set[p] < ns:
                if ns > len(synergies[p].heroIndexes):
                    return
                s_num = ns - synergy_set[p]
                if s_num + hl > n:
                    return
                h_list = synergies[p].heroIndexes.copy()
                lowest = getLowest(hero_list, hl, p)
                # removeLowIndex(lowest, h_list)
                removeSame(hero_list, hl, h_list)
                selectAddDo(h_list, s_num, perfectCombSmart, n, synergy_set, first, hero_list, hl)
                break


def findAllSameSynergyHero():
    for index1 in range(len(heroes)):
        for index2 in range(index1 + 1, len(heroes)):
            if heroes[index1].jobIndexes == heroes[index2].jobIndexes:
                added = 0
                for group in same_list:
                    if index1 == group[-1]:
                        group.append(index2)
                        added = 1
                        break
                if added == 0:
                    same_list.append([index1, index2])
            index2 += 1
        index1 += 1
    return same_list


def start():
    # print(synergies[12].name, synergies[12].nums)

    i = 0
    for s in synergies:
        s.initialHeroIndexes()
        # if len(s.heroIndexes) == 0:
        #     synergies.remove(s)
        # else:
        print(i, s.name, s.heroIndexes, s.nums)
        i += 1
    print("------------------------------------")
    i = 0
    for h in heroes:
        h.initialJobIndexes(synergies)
        print(i, h.name, h.jobIndexes)
        i += 1


if __name__ == '__main__':
    # downloadHtmlAndSave("https://lol.qq.com/tft/#/champion", "yundinHero.html")
    # downloadHtmlAndSave("https://lol.qq.com/tft/#/synergy", "yundinSynergy.html")
    heroes = [Hero]
    heroes.pop()
    synergies = [Synergy]
    synergies.pop()
    with open("yundinHero.html", 'r') as f:
        page_content = f.read()
        extractHeros(page_content)
    with open("yundinSynergy.html", "r") as f:
        html = f.read()
        extractSynergies(html)
    start()
    same_list = []
    findAllSameSynergyHero()
    # print(same_list)
    perfect_num = 8
    # perfectComb(perfect_num)
    # C(4, [0]*4)
    perfectCombSmart(n=perfect_num)
    print(len(perfect_list))
    # printIfPerfect([0,1,3,14,31,49,53,56], {})
    # ii = 0
    # for h in heroes:
    #     print(ii, h.name)
    #     ii+=1

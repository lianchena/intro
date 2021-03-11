###PASS7

# ie - 6 marks
# for v in lst[:i]:
#    lst[k] <= v

# for q in range(:i+1):
#    lst[k] <= lst[q]



def getProduct(myList):
    value = 1
    for i in range(len(myList)):
        #invariant1: at the end of the i th iteration, value
        #represents the product of all the items in myList[i]
        value *= myList[i]
        #invariant2: : at the end of the i th iteration, value
        #represents the product of all the items in myList[i+1]
    return value
# getProduct([4,3,-2])
# step 1: what function does - multipies all numbers in list
# step 2: pattern in key variable
#           key variable - value
#           before/after condition - value = 4; value = 12

def getMaxIndex(myList, start, stop):
    max_index = start
    for i in range(start+1, stop):
        #invariant1:
        if myList[i] > myList[max_index]:
            max_index = i
        #invariant2:
    return max_index

###lec16
def unique_values(lst):
    '''
    best-case O(n)
    worst-case O(n**2)
    '''
    n = len(lst)
    res = []
    for i in range(n):
        last_occurrence = True
        for j in range(i+1, n):
            if lst[i] == lst[j]:
                last_occurrence = False
                break
        if last_occurrence:
            res += [lst[i]]
    return res

def unique_values(lst):
    '''
    best-case O(n**2)
    worst-case O(n**2)
    '''
    n = len(lst)
    res = []
    for i in range(n):
        if lst[i] not in lst[i+1:]:
            res += [lst[i]]
    return res

# 1 - another input: pre-sorting
def unique_values(lst):
    '''
    worst-case O(nlogn)
    '''
    n = len(lst)
    lst = mergesort(lst)
    res = []
    for i in range(n):
        if i == n-1 or lst[i] != lst[i + 1]:
            res += [lst[i]]
    return res
# 2 - another represnetation: anagrams
def order_invariant_indexing(words):
    res = []
    for i in range(len(words)):
        res += [("".join(sorted(words[i])),words[i])]
    return res
      
def anagrams(words):
    augmented = order_invariant_indexing(words)
    ordered = sorted(augmented)
    
    res = []
    i = 0
    while i < len(ordered):
        res.append([])
        key = ordered[i][0]
        while i < len(ordered) and \
              ordered[i][0] == key:
            res[-1].append(ordered[i][1])
            i = i + 1
    return res
# 3 - another problem: least common multiple
def gcd(a,b):
    if b == 0:
        return a
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a*b/gcd(a,b)



###lec15
def reachable(graph, s):
    '''
    Input: graph 𝐺 = 𝑉, 𝐸 and vertex 𝑠 ∈ 𝑉
    Output: all vertices 𝑣 connected to 𝑠
    (there is path between 𝑣 and 𝑠) 
    '''
    res = [s]
    for v in neighbours(s, graph):
        if v not in res:
            res += reachable(graph, v)
    return res
def reachable(graph, s, visited=None):
    if visited is None: visited = [False]*len(graph)
    res = [s]
    visited[s] = True
    for v in neighbours(s, graph):
        if not visited[v]:
            res += reachable(graph, v, visited)
    return res
def reachable(graph, s):
    visited = []
    boundary = [s]
    while len(boundary) > 0:
        v = boundary.pop()
        visited += [v]
        for w in neighbours(v, graph):
            if w not in visited and w not in boundary:
                boundary.append(w)
    return visited

#Decrease-and-Conquer Power (with a stack)
def power(x, n):
    if n == 0:
        return 1
    else:
        value = power(x, n//2)
        if n % 2 == 0:
            value = value*value
        else:
            value = value*value*x
        return value

def power(x, n):
    stack = []
    while n > 0:
        stack.append(n)
        n = n // 2
    res = 1
    while len(stack)>0:
        n = stack.pop()
        if n % 2 == 0:
            res = res * res
        else:
            res = res * res * x
    return res

# shortest path problem
def dfs_traversal(graph, s):
    visited = []
    boundary = [s]
    while len(boundary) > 0:
        v = boundary.pop()
        visited += [v]
        for w in neighbours(v, graph):
            if w not in visited and w not in boundary:
                boundary.append(w)
    return visited

def bfs_traversal(graph, s):
    visited = []
    boundary = [s]
    while len(boundary) > 0:
        v = boundary.pop(0)
        visited += [v]
        for w in neighbours(v, graph):
            if w not in visited and w not in boundary:
                boundary.append(w)
    return visited

# queues
from collections import deque
def bfs_traversal(graph, s):
    visited = []
    boundary = deque([s])
    while len(boundary) > 0:
        v = boundary.popleft()
        visited += [v]
        for u in neighbours(v, graph):
            if u not in visited and u not in boundary:
                boundary.append(u)
    return visited
'''
    keeping track of distances
def bfs_distances(graph, s):
    dists = [inf] * len(graph)
    dists[s] = 0
    visited = []
    boundary = deque([s])
    while len(boundary) > 0:
        v = boundary.popleft()
        visited += [v]
        for u in neighbours(v, graph):
        if u not in visited and u not in boundary:
            boundary.append(u)
            dists[u] = dists[v] + 1
    return dists
'''
    
###lec14
#Fibonacci numbers
def fib(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return test(n-2)+test(n-1)

def merge(l1, l2):
    '''
    n = n1 + n2
    best-case O(n)
    worst-case O(n)
    '''
    res = []
    n1, n2 = len(l1), len(l2)
    i, j = 0, 0
    while i < n1 and j < n2:
        if l1[i] <= l2[j]:
            res += [l1[i]]
            i += 1
        else:
            res += [l2[j]]
            j += 1
    return res + l1[i:] + l2[j:]
def merge_sort(ls):
    n = len(ls)
    if n <= 1:
        return ls
    else:
        sub1 = mergesort(ls[:n//2])
        sub2 = mergesort(ls[n//2:])
        return merge(sub1, sub2)

# Towers of Hanoi
def start_state(disk):
    res = []
    for i in range(disk,0,-1):
        res += [i]
    return [res, [], []]
def next_state(start,to,state):
    state[to] += [state[start].pop()]
    return state
def aux(i,j):
    for res in range(3):
        if i != res and j != res:
            return res

def hanoi(n, i, j, state):
    '''
    worst-case O(n * 2**n)
    '''
    """
    I: num n of disks to move from needle i to needle j
    in state such that state[i][-n]<state[j][-1]
    and state[i][-n]<state[aux(i,j)][-1]
    O: intermediate states when carrying out input task
    """
    if n == 1:
        return [(next_state(i, j, state))]
    else:
        sub1 = hanoi(n - 1, i, aux(i, j), state)
        nxt = next_state(i, j, sub1[-1])
        sub2 = hanoi(n - 1, aux(i, j), j, nxt)
    return sub1 + [nxt] + sub2


###lec13
'''
def insertion_merge(l1, l2):
    res = l1 + l2
    n1, n2 = len(l1), len(l2)
    for i in range(n1, n1+n2):
        insert(i, res)
    return res
'''

def merge(l1, l2):
    '''
    n = n1 + n2
    best-case O(n)
    worst-case O(n)
    '''
    res = []
    n1, n2 = len(l1), len(l2)
    i, j = 0, 0
    while i < n1 and j < n2:
        if l1[i] <= l2[j]:
            res += [l1[i]]
            i += 1
        else:
            res += [l2[j]]
            j += 1
    return res + l1[i:] + l2[j:]

def mergesort(ls):
    '''
    n = len(ls)
    best-case O(nlog(n)) 
    worst-case O(nlog(n)) 
    '''
    k, n = 1, len(ls)
    while k < n:
        nxt = []
        for a in range(0, n, 2*k):
            b, c = a + k, a + 2*k
            nxt += merge(ls[a:b], ls[b:c])
        ls = nxt
        k = 2 * k
    return ls



### lec 12
def sequential_search(v, seq):
    '''
    best-case O(1)
    worst-case O(n)
    #I) v in seq[i:] or v not in seq
    # v not in seq[:i]
    '''
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] == v:
            return i
        i += 1
    return None

def probing_search(v, seq):
    a, b = 0, len(seq)-1
    c = b // 2
    if seq[c] == v:
        return c
    elif v < seq[c]:
        b = c - 1
    else:
        a = c + 1
    i = a
    while a <= i <= b:
        if seq[i] == v:
            return i
        i += 1
    return None

def binary_search(v, seq):
    '''
    n = len(lst)
    best-case O(1)
    worst-case O(log(n))  (which is log2(n))
    '''
    a = 0
    b = len(seq) - 1
    while a <= b:
        c = (a + b) // 2
        if seq[c] == v:
            return c
        elif seq[c] > v:
            b = c - 1
        else:
            a = c + 1
    return None

def gcd_euclid(a, b):
    '''
    in all cases: problem is at least decreased by a rate r of 5/4
    n = abs(a)+abs(b)
    best-case O(1)
    worst-case O(log(n))  (which is logr(n))
    '''
    '''
    #I: gcd(a,b)==gcd(a0,b0)
    #I: gcd(a,b)==gcd(a0,b0)
    #PRC: a,b==a0,b0 (original input)
    #EXC: b==0
    #POC: a==gcd(a,b)==gcd(a0,b0)
    '''
    """
    Input : integers a and b such that not a==b==0
    Output: the greatest common divisor of a and b
    """
    while b != 0:
        a, b = b, a % b
    return a


###lec11
def gcd_brute_force(m, n):
    """Input : integers m and n such that not n==m==0
    Output: the greatest common divisor of m and n
    """
    x = min(m, n)
    while not (m % x == 0 and n % x == 0):
        x = x - 1
    return x

def gcd_euclid(m, n):
    """
    Input : integers m and n such that not n==m==0
    Output: the greatest common divisor of m and n
    """
    while n != 0:
        m, n = n, m % n
    return m

###lec9
graph = \
    [[0,1,0,0,0,0,0,0,0],
    [1,0,1,1,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,0],
    [0,0,0,1,0,1,0,0,0],
    [0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,1,1,0]]
def neighbours(i, g):
    '''I: vertex i, graph g
    O: neighbours of i
    For example:
    >>> neighbours(5, graph) 
    [3, 4]
    '''
    n = len(g)
    res = []
    for j in range(n):
        if g[i][j]==1:
            res.append(j)
    return res

#Prim's
def spanning_tree(graph):
    '''Input : adjacency matrix of graph
    Output: adj. mat. of spanning tree of graph'''
    n = len(graph)
    tree = empty_graph(n)
    conn = {0}
    while len(conn) < n:
        found = False
        for i in conn:
            for j in range(n):
                if j not in conn and graph[i][j]==1:
                    tree[i][j] = 1
                    tree[j][i] = 1
                    conn = conn.add(j)
                    found = True
                    break
            if found:
                break
    return tree


#factor out extention(decomposition)
'''
def extension(c, g):
    'I: connect. vertices, graph
    O: extension edge (i, j)'
    n = len(g)
    for i in c:
        for j in range(n):
            if j not in c \
            and g[i][j]:
                return i, j
n = len(graph)
tree = empty_graph(n)
conn = {0}
while len(conn) < n:
    i, j = extension(conn, graph) 
    tree[i][j] = 1
    tree[j][i] = 1
    conn.add(j)
return tree
'''



###lec8
#selection
def min_index(lst):
    temp = min(lst)
    for idx in range(len(lst)):
        if temp == lst[idx]:
            res = idx
    return res
def selection_sort(lst):
    '''
    accepts : list lst of length n of comp. elements
    post-cond: lst has same elements as on call but
    for all i in range(1,n), lst[i-1]<=lst[i]
    '''
    for i in range(len(lst)):
        j = min_index(lst[i:]) + i
        lst[i], lst[j] = lst[j], lst[i]
    return lst


#insertion
def insert(k, lst):
    '''
    accepts: list lst of length n>k>=0 of comp. elements
    such that lst[:k] is sorted
    postcon: lst[:k+1] is sorted
    '''
    j = k
    while j > 0 and lst[j - 1] > lst[j]:
        lst[j - 1], lst[j] = lst[j], lst[j - 1]
        j = j - 1
def insertion_sort(lst):
    '''
    accepts : list lst of length n of comp. elements
    post-cond: lst has same elements as on call but
    for all i in range(1,n), lst[i-1]<=lst[i]
    '''
    for i in range(1, len(lst)):
        insert(i, lst)
    return lst

###lec7
'''
def swap(x, y):
    tmp = x
    x = y
    y = tmp
    return x,y'''

def swap(x, y):
    return y, x

###lec6&7
''''
def list_from_file(filename):
    file = open(filename)
    res = []
    for line in file:
        res = res + [line.strip()]
    file.close()
    return res

def list_from_file(fname, num=False):
    file = open(fname)
    rs = []
    for l in file:
        if num:
            rs = rs+[float(l.strip())]
        else:
            rs = rs+[l.strip()]
    file.close()
    return rs

def list_from_file(fname, typ='str'):
    file = open(fname)
    rs = []
    for l in file:
        if typ=='float':
            rs = rs+[float(l.strip())]
        elif typ=='int':
            rs = rs+[int(l.strip())]
        else:
            rs = rs+[l.strip()]
    file.close()
    return rs '''

def list_from_file(filename, typ=str):
    file = open(filename)
    res = []
    for line in file:
        res = res + [typ(line.strip())]
    file.close()
    return res

def as_float(num, seq):
    res = []
    for i in range(len(seq)):
        if type(seq[i]) == 'float':
            res = res + [float(seq[i])]
        else:
            res = seq[i]
    return res
    
def table_from_file(filename, num_cols=[]):
    lines = list_from_file(filename)
    cols = lines[0].split(',')[1:]
    ids, tab = [], []
    for i in range(1, len(lines)):
        entries = lines[i].split(',')
        ids = ids + [entries[0]]
        row = (num_cols, entries[1:])
        tab = tab + [row]
    return tab, cols, ids



def scaled(row, alpha):
    
    """
    Input : list with numeric entries (row), scaling factor (alpha)
    Output: new list (res) of same length with res[i]==row[i]*alpha
    For example:
    >>> scaled([1, 4, -1], 2.5)
    [2.5, 10.0, -2.5]
    """
    res = []
    for i in range(len(row)):
        res = res + [alpha*row[i]]
    return res


def nutrients(food, quantity):
    for i in range(len(foods)):
        if foods[i]==food:
            nutr_100g = nutr_vals[i]
            return scaled(nutr_100g, quantity/100)
        
def sum_of_rows(r1, r2):
    """
    Input : two lists (r1, r2) with same number of numeric entries
    Output: new list (res) of same length with res[i]==r1[i]+r2[i]
    for all i in range(len(r1))
    For example:
    >>> sum_of_rows([100, -4, 10], [0, 3.5, -10])
    [100, -0.5, 0]
    """
    res = []
    for i in range(len(r1)):
        res += [r1[i] + r2[i]]
    return res

        
def intake_per_day(food_diary):
    days, intake = [], []
    for day, food, quantity in food_diary:
        nutr = nutrients(food, quantity)
        if day not in days:
            days = days + [day]
            intake = intake + [nutr]
        else:
            intake[-1] = sum_of_rows(intake[-1], nutr)
    return intake, days




def as_str(lst):
    """Converts lst of objects to list of strings.""" 
    res = []
    for x in lst:
        res.append(str(x))
    return res

def table_to_file(vals, cols, ids, filename):
    """
    Writes a table with column names and ids to csv file.
    Input : table (vals) with column names (cols), and
    row ids (ids), name of output file (filename)
    Output: None; writes table to file
    """
    file = open(filename, 'w')
    header = 'id,' + ','.join(cols) + '\n'
    file.write(header)
    for i in range(len(vals)):
        line = str(ids[i]) + ',' + ','.join(as_str(vals[i])) + '\n'
        file.write(line)
    file.close()



nvals, nutr_cols, foods = table_from_file('nutr_table.csv' , range(7))
food_diary, _, _ = table_from_file('food_diary.csv', [3])
intake, days = intake_per_day(food_diary)
table_to_file(intake, nutrient_names, days, 'intake_per_day.csv')




###lec5
def times_eaten(food, eaten_foods):
    """
    Input : specific food, list of eaten foods
    Output: number of times food appears in eaten_foods
    """
    res = 0
    for f in eaten_foods:
        if f == food: res = res + 1
    return res

def violated_diet(eaten_foods, forbidden_foods):
    for food in eaten_foods:
        if food in forbidden_foods: return True
    return False

def quantity_eaten(food, eaten_foods, eaten_quantities):
    """
    Input : specific food, list of eaten foods, 
    list of eaten quantities
    Output: total quantity of specific food eaten
    """
    res = 0
    for i in range(len(eaten_foods)):
        if eaten_foods[i] == food:
            res = res + eaten_quantities[i]
    return res

def have_common_element(s1, s2):
    for a in s1:
        for b in s2:
            if a==b:
                return True
    return False

def pyramidal(n):
    """
    Input: an integer n
    Output: number of cannonballs in pile of height n

    For example:
    >>> pyramidal(1)
    1
    >>> pyramidal(3)
    14
    """
    count = 0
    for k in range(1, n+1):
        count = count + k**2
    return count

###lec4
def sum_of_first_n_ints(n):
    """
    Input : positive integer n
    Output: sum of pos. integers up to n"""
    i = 1 #iteration variable
    res = 0 #accumulation variable
    while i <= n:
        res = res + i
        i = i + 1
    return res

def pi_approximation(eps):
    """
    Input : positive float eps (accuracy)
    Output: approximation x to pi with abs(x-pi)<=eps
    """
    i = 0 #iteration variable
    x = 0 #candidate solution 
    while True:
        i += 1
        z = x
        x = x + (-1)**(i+1)*4/(2*i-1)
    '''if abs(x - z) <= eps:

     break x'''
    pass

def gcd_brute_force(m, n):
    x = min(m, n)
    while not (m % x == 0 and n % x == 0):
        x = x - 1
    return x

def gcd_brute_force(m, n):
    x = min(m, n)
    while (m % x != 0) or (n % x != 0):
        x = x - 1
    return x


#Euclid's Algorithm
def gcd(m, n):
    """
    Input : integers m and n such that not n==m==0
    Output: the greatest common divisor of m and n
    """
    while n != 0:
        r = m % n
        m = n
        n = r
    return m


###lec3
def price_after_gst(gross_price):
    """
    Input : gross price of product
    Output: sales price of product (incorporating GST)
    """
    gst = gross_price * gst_rate
    price = gross_price + gst
    return price


gst_free_products = ['bread', 'peach', 'tea']

def price_after_gst(gross_price, product):
    if product in gst_free_products:
        gst = 0
    else:
        gst = gross_price * gst_rate
    return gross_price + gst

def customer_price(purch_price, product, best_before_days):
    base_price = purch_price + purch_price*0.1
    if best_before_days <= 1:
        reduction = base_price*0.6
    elif best_before_days <= 4:
        reduction = base_price*0.3
    else:
        reduction = 0
    gross_price = base_price - reduction
    return round(price_after_gst(gross_price, product), 2)


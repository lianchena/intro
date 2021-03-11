def mean_grade(stud_id, results):
    for i in range(len(results)):
        if results[i][2] == stu_id:
            add += results[i][3]
            num += 1
        res = add/num
        return res

def simplified(string):
    if len(string) == 0 or len(string) == 1:
        return string
    elif string[0] == string[1]:
        return string[1]+simplified(string[2:])


# invariants
#INV: res = max(lst[:i])
#EXC: i == len(lst)
#POC: res = max(lst[:len(lst)])


#selection
def min_index(lst):
    '''
    best-case O(n)
    worst-case O(n)
    '''
    '''
    #I: for all j in range(i): lst[k]<=lst[j]
    #I’: for all j in range(i+1): lst[k]<=lst[j]
    #EXC: i = n-1
    #POC: for j in range(n): lst[k]<=lst[j]
    '''
    k = 0
    for i in range(1, len(lst)):
        if lst[i] < lst[k]:
            k = i
    return k

def selection_sort(lst):
    '''
    worst-case O(n**2)
    '''
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
    best-case O(1)
    worst-case O(n)
    '''
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
    best-case O(n)
    worst-case O(n**2)
    '''
    '''
    accepts : list lst of length n of comp. elements
    post-cond: lst has same elements as on call but
    for all i in range(1,n), lst[i-1]<=lst[i]
    '''
    '''
    #I: lst[i] sorted
    #I’: lst[i+1] sorted
    #EXC: i = n-1
    #POC: lst[:n] sorted
    '''
    for i in range(1, len(lst)):
        insert(i, lst)
    return lst






def extension(con, g):
    for i in con:
        for j in range(len(g)):
            if j not in con and g[i][j]:
                return i, j
def spanning_tree(graph):
    '''
    #I1: con is connected in tree
    #I2: tree does not contain cycle

    #EXC: len(con)==len(graph)
    #POC: tree is spanning tree of graph
    '''
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











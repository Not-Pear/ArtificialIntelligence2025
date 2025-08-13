import numpy as np
import heapq
import random

# GEOMETRY CLASS
class Geometry:
    count = 0
    def __init__(self, name = "Shape", points = None):
        self.name = name
        # name is string that is a name of gemoetry
        self.points = points
        # points is a list of tuple points = [(x0, y0), (x1, y1), ...]
        Geometry.count += 1

    def calculate_area(self):
        return 0.0

    def get_name(self):
        return self.name

    @classmethod
    def count_number_of_geometry(cls):
        # DONE: Your task is to implement the class method
        # to get the number of instance that have already created
        return Geometry.count
        


# TRIANGLE CLASS
class Triangle(Geometry):
    def __init__(self, a, b, c):
        # a, b, c are tuples that represent for 3 vertices of a triangle
        # DONE: Your task is to implement the constructor
        self.a = a
        self.b = b
        self.c = c
        super(Triangle, self).__init__(name="Triangle", points=[a,b,c])

    def calculate_area(self):
        #DONE: Your task is required to implement a area function
        # Formula: 0.5 |x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)|
        area = 0.5 * abs(self.a[0] * (self.b[1] - self.c[1]) + self.b[0] * (self.c[1] - self.a[1]) + self.c[0] * (self.a[1] - self.b[1]))
        return area


# RECTANGLE CLASS
class Rectangle(Geometry):
    def __init__(self, a, b):
        # a, b are tuples that represent for top and bottom vertices of a rectangle
        # DONE: Your task is to implement the constructor
        self.a = a
        self.b = b
        super(Rectangle, self).__init__(name="Rectangle", points=[a, b])

    def calculate_area(self):
        #DONE: Your task is required to implement a area function
        width = abs(self.b[0] - self.a[0])
        height = abs(self.b[1] - self.a[1])
        
        area = width * height
        return area


# SQUARE CLASS
class Square(Rectangle):
    def __init__(self, a, length):
        # a is a tuple that represent a top vertex of a square
        # length is the side length of a square
        # DONE: Your task is to implement the constructor
        self.a = a
        x, y = a
        b = (x + length, y - length)
        self.length = length
        super(Square, self).__init__(a, b)
        self.name = "Square"

    def calculate_area(self):
        #DONE: Your task is required to implement a area function
        area = self.length**2 # a^2
        return area


# CIRCLE CLASS
class Circle(Geometry):
    def __init__(self, o, r):
        # o is a tuple that represent a centre of a circle
        # r is the radius of a circle
        # DONE: Your task is to implement the constructor
        self.o = o
        self.r = r
        super(Circle, self).__init__(name="Circle", points=[o])

    def calculate_area(self):
        #DONE: Your task is required to implement a area function
        area = np.pi * (self.r ** 2)
        return area


# POLYGON CLASS
class Polygon(Geometry):
    def __init__(self, points):
        # points is a list of tuples that represent vertices of a polygon
        # DONE: Your task is to implement the constructor
        self.points = points
        self.n = len(points)
        super(Polygon, self).__init__(points=points)

    def calculate_area(self):
        #DONE: Your task is required to implement a area function
        # Formula: Shoelace/Gauss's area 
        area = 0
        for i in range(self.n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % self.n] # Wraps around to the first vertex
            area += x1 * y2 - y1 * x2
        
        return abs(area) / 2



#**************************** IGNORE ****************************# 
def test_geometry():
    ## Test cases for Problem 1

    triangle = Triangle((0, 1), (1, 0), (0, 0))
    print("Area of %s: %0.4f" % (triangle.name, triangle.calculate_area()))

    rectangle = Rectangle((0, 0), (2, 2))
    print("Area of %s: %0.4f" % (rectangle.name, rectangle.calculate_area()))

    square = Square((0, 0), 2)
    print("Area of %s: %0.4f" % (square.name, square.calculate_area()))

    circle = Circle((0, 0), 3)
    print("Area of %s: %0.4f" % (circle.name, circle.calculate_area()))

    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    print("Area of %s: %0.4f" % (polygon.name, polygon.calculate_area()))
#**************************** IGNORE ****************************#

def matrix_multiplication(A, B):
    # DONE: Your task is to required to implement
    # a matrix multiplication between A and B
    m_A, n_A = A.shape
    n_B, p_B = B.shape
    
    if n_A != n_B:
        return -1
    

    C = np.zeros((m_A, p_B), np.float64)

    for i in range(m_A): # Row of matrix A
        for j in range(p_B): # Column of Matrix B
            for k in range (n_A): # Column of matrix A and Row of matrix B
                C[i,j] += A[i,k] * B[k,j]
            
    return C



#**************************** IGNORE ****************************#
def test_matrix_mul():
    ## Test cases for matrix multplication ##

    for test in range(10):
        m, n, k = random.randint(3, 10), random.randint(3, 10), random.randint(3, 10)
        A = np.random.randn(m, n)
        B = np.random.randn(n, k)
        assert np.mean(np.abs(A.dot(B) - matrix_multiplication(A, B))) <= 1e-7, "Your implmentation is wrong!"
        print("[Test Case %d]. Your implementation is correct!" % test)
#**************************** IGNORE ****************************#


def recursive_pow(A, n):
    # DONE: Your task is required implementing
    # a recursive function
    size, _ = A.shape
    if n == 0:
        return np.identity(size, np.float64)
    elif n == 1:
        return A
    elif n == 2:
        return A.dot(A)
    else:
        r = n % 2
        k = n // 2 # Integer division instead of float division `/`
        return (recursive_pow(A, k).dot(recursive_pow(A, k))).dot(recursive_pow(A, r))


def iterative_pow(A, n):
	# DONE: Your task is required implementing
    # a iterative function
    cumulative_matrix = A
    for i in range(n - 1):
        cumulative_matrix = cumulative_matrix.dot(A)
    
    return cumulative_matrix


#**************************** IGNORE ****************************#
def test_pow():
    ## Test cases for the pow function ##

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Recursive: A^{} = {}".format(n, recursive_pow(A, n)))

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Iterative: A^{} = {}".format(n, recursive_pow(A, n)))


def test_pow_clearly():
    n = 2
    A = np.array([[1, -3], [2, 5]])
    print("Starting Array: \n", A)
    print("Recursive: A^{} =\n {}".format(n, recursive_pow(A, n)))
    print("")
    print("Iterative: A^{} =\n {}".format(n, iterative_pow(A, n)))

#**************************** IGNORE ****************************#


def get_A():
    # DONE: Find a matrix A
    # You have to return in the format of numpy array
    return np.array([[1.,1.], [1.,0.]])

def fibo(n):
    # DONE: Calcualte the n'th Fibonacci number
    A = get_A()
    F1 = np.array([[1.],[1.]])
    FN = recursive_pow(A, n - 1).dot(F1)
    return int(FN[0][0])

# Naive solution.
def f(n, k):
    # DONE: Calculate the n'th number of the recursive sequence
    if (n < k):
        return 1
    else: 
        ret = 0
        for i in range (1, k + 1):
            ret += f(n-i, k)
        return ret


#**************************** IGNORE ****************************#
def test_fibonacci():
    ## Test Cases for Fibonacci and Recursive Sequence ##

    a, b = 1, 1
    for i in range(2, 10):
        c = a + b
        assert (fibo(i) == c), "You implementation is incorrect"
        print("[Test Case %d]. Your implementation is correct!. fibo(%d) = %d" % (i - 2, i, fibo(i)))
        a = b
        b = c

    for n in range(5, 11):
        for k in range(2, 5):
            print("f(%d, %d) = %d" % (n, k, f(n, k)))
#**************************** IGNORE ****************************#


def recursiveDFS(x, y, A, visited, path):
    M = A.shape[0]
    N = A.shape[1]
    visited[x][y] = 1
    if (x == M - 1 and y == N - 1):
        print("(0, 0)", end = "")
        for (u, v) in path:
            print(" -> (%d, %d)" % (u, v), end ="")
        print()
        exit(0)
    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        u = x + dx
        v = y + dy
        if (0 <= u) and (u < M) and (0 <= v) and (v < N) and (A[u][v] != 0) and visited[u][v] == 0:
            path.append((u, v))
            recursiveDFS(u, v, A, visited, path)
            path.pop()

def DFS(A):
    print("DFS")
    # A is a mxn matrix
    recursiveDFS(0, 0, A, np.zeros_like(A), [])



def BFS(A):
    # A is a mxn matrix
    print("BFS")
    M = A.shape[0]
    N = A.shape[1]
    visited = np.zeros_like(A)
    prevx = np.ones_like(A) * -1
    prevy = np.ones_like(A) * -1
    queue = [(0, 0)]
    visited[0][0] = 1
    while (len(queue) > 0):
        x, y = queue.pop(0)
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            u = x + dx
            v = y + dy
            if (0 <= u) and (u < M) and (0 <= v) and (v < N) and (A[u][v] != 0) and visited[u][v] == 0:
                prevx[u][v] = x
                prevy[u][v] = y
                queue.append((u, v))
                visited[u][v] = 1
    path = []
    x, y = M - 1, N - 1
    while (x != 0 or y != 0):
        u, v = prevx[x][y], prevy[x][y]
        path.append((u, v))
        x, y = u, v
    for u, v in path[::-1]:
        print("(%d, %d) -> " % (u, v), end="")
    print("(%d, %d)" % (M - 1, N - 1))

# Dijkstra's Algorithm

def findMinimum(A):
    print("Find Minimum")
    # A is a mxn matrix
    M = A.shape[0]
    N = A.shape[1]
    visited = np.zeros_like(A)
    prevx = np.ones_like(A) * -1
    prevy = np.ones_like(A) * -1
    cost = np.ones_like(A) * 100000
    queue = []
    heapq.heappush(queue, (A[0][0], (0, 0)))
    cost[0][0] = A[0][0]
    while (len(queue) > 0):
        c, (x, y) = heapq.heappop(queue)
        visited[x][y] = 1
        if (cost[x][y] != c):
            continue
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            u = x + dx
            v = y + dy
            if (0 <= u) and (u < M) and (0 <= v) and (v < N) and (A[u][v] != 0) and visited[u][v] == 0 and cost[x][y] + A[u][v] < cost[u][v]:
                prevx[u][v] = x
                prevy[u][v] = y
                cost[u][v] = cost[x][y] + A[u][v]
                heapq.heappush(queue, (cost[u][v], (u, v)))
    path = []
    x, y = M - 1, N - 1
    while (x != 0 or y != 0):
        u, v = prevx[x][y], prevy[x][y]
        path.append((u, v))
        x, y = u, v
    print("Cost: %d" % cost[M-1][N-1])
    for u, v in path[::-1]:
        print("(%d, %d) -> " % (u, v), end="")
    print("(%d, %d)" % (M - 1, N - 1))


#**************************** IGNORE ****************************#
def test_bfs_dfs_find_minimum():
    ## Test Cases for BFS, DFS, Find Minimum ##
    A = np.array([[1, 1, 1, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]])

    findMinimum(A)

    BFS(A)

    DFS(A)

    A = np.array([[1, 1, 1, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 2], [1, 1, 0, 2, 1], [1, 1, 0, 2, 1]])


## Testing Your Code

test_geometry()
test_matrix_mul()
# test_pow()
test_pow_clearly()
test_fibonacci()
test_bfs_dfs_find_minimum()


#**************************** IGNORE ****************************#



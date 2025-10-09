from sympy import symbols, diff, integrate, sin, exp, log

x = symbols('x')

# 示例1：求导
f = x**3 + sin(x) * exp(x)
df = diff(f, x)
print("导数:", df)  # 输出: 3*x**2 + exp(x)*sin(x) + exp(x)*cos(x)

# 示例2：积分
g = log(x) / x
integral = integrate(g, x)
print("积分:", integral)  # 输出: log(x)**2 / 2


class Symbol:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def diff(self, var):
        return 1 if self == var else 0  # 变量求导为1，否则为0


class Add:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} + {self.right})"

    def diff(self, var):
        return Add(self.left.diff(var), self.right.diff(var))  # 和的导数=导数的和


class Mul:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} * {self.right})"

    def diff(self, var):
        # 乘积法则: (uv)' = u'v + uv'
        return Add(
            Mul(self.left.diff(var), self.right),
            Mul(self.left, self.right.diff(var))
        )


class Sin:
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f"sin({self.arg})"

    def diff(self, var):
        # 链式法则: sin(f(x))' = cos(f(x)) * f'(x)
        return Mul(Cos(self.arg), self.arg.diff(var))


class Cos:  # 新增的Cos类
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f"cos({self.arg})"

    def diff(self, var):
        # 链式法则: cos(f(x))' = -sin(f(x)) * f'(x)
        return Mul(Mul(Symbol('-1'), Sin(self.arg)), self.arg.diff(var))


# 示例使用
x = Symbol('x')
expr = Mul(x, Sin(x))  # 表示 x*sin(x)
print(expr)  # 输出: (x * sin(x))
print(expr.diff(x))  # 输出: (1 * sin(x)) + (x * cos(x)*1)
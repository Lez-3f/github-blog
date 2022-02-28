# **Python基础语法**

## 基本输入输出


使用`print()`输出


```python
print("Hello, world");

# 格式化输出
s1 = "Python"
s2 = "C"
print("Hello, {0}! Goodbye, {1}!".format(s1, s2))
```

使用`input()`输入

一定要注意，input后的数据一定是字符串，在使用的时候要转换成需要的类型！！！


```python
s1 = input("Enter");
s2 = input();

print("{0} likes {1}. {1} doesn't like {0}.".format(s1, s2));
```

## 判断语句



python使用缩进来确定代码结构的，所以一定要学会使用`Tab`键！


```python
import random
score = input("输入您的分数:") 
teacher_mood = random.randint(0, 9);  # 生成0-9的随机整数
score = int(score)
grade = "F"
if score >= 98:
    grade = "A+"
elif score >= 90 and score < 98:
    grade = "A"
elif score >= 80 and score < 90:
    grade = "B"
elif score >= 70 and score < 85:
    grade = "C"
elif (score >= 60 and score < 70) or (score >= 30 and teacher_mood > 4):
    grade = "D"
else:
    grade = "F"

print("您的等级是:" + grade)
```

## 循环语句


### for循环


```python
# 输出1-5的平方
for i in range(1, 6):
    print(i ** 2)

```

### while循环


```python
n = 5;
while n > 0:
    print(n ** 2)
    n -= 1
```

## 数据类型和变量



### 基本数据类型

整数，浮点数，字符串, 复数

### 基本数据结构
### list
列表：list是一种有序的集合，可以随时**添加**和**删除**其中的元素
### tuple
元组：tuple和list非常类似，但是tuple一旦初始化就**不能修改**
### dict
字典：使用**键-值**（key-value）存储，具有极快的查找速度。
### set
集合：无重复元素


```python
print(2 ** 100000) #整数长度不受限制
```


```python
print(1 / 13548413511)
```


```python
print("Hello" + 'World\n')
```


```python
print((23 + 15j) * (13 + 221j))
```


```python
l = ["Calculous-A", "LinearAlgebra","BasicPhysic"]  # 定义列表
print(l)
l.append("PythonPragraming") # 插入元素
print(l)
l.remove("LinearAlgebra") # 删除元素
print(l)
l[0] = "Calculous-B" # 访问列表
print(l)

# 列表生成器
l2 = [i ** 2 for i in range(1, 6) if i % 2 == 1]
print(l2)
```


```python
t = ("TsinghuaQuan", "Basketball")
# t[0] = "Lalacao"
# t.remove("TsinghuaQuan") 
```


```python
d = {"Calculous-A" : "6A101", "LinearAlgebra" : "5201", "BasicPhysic": "3213"}
print(d)
print(d["Calculous-A"]) # 访问
# print(d["PythonPragraming"])
d["PythonPragraming"] = "3200" # 插入
print(d)
d["Calculous-A"] = "6A102" # 修改
print(d)
d.pop("LinearAlgebra") # 剔除
print(d)
```


```python
s = set(l)
print(s)
s.add("Calculous-A") # 插入
print(s)
s.remove("Calculous-A") # 删除
print(s)

print("MathematicalAnalysis" in s) # 查询

s = set([1,1, 2, 2, 3,3,3,4,4,4,4,4,4,4,4,4,4,4,4]) # 过滤重复元素
print(s)
```

## 函数定义


```python
import random
def getGrade(score):
    if not isinstance(score, int):
        raise TypeError('bad operand type') #参数类型检查
    # 参数正确
    teacher_mood = random.randint(0, 9);  # 生成0-9的随机整数
    if score >= 98:
        return "A+", 4.0
    elif score >= 90 and score < 98:
        return "A", 4.0
    elif score >= 80 and score < 90:
        return "B", 3.0
    elif score >= 70 and score < 85:
        return "C", 2.0
    elif (score >= 60 and score < 70) or (score >= 30 and teacher_mood > 4):
        return "D", 1.0
    else:
        return "F", 0.0 # 返回tuple

grade, point = getGrade(95)
print(grade, point)
print(getGrade(58))
#getGrade("hello")
```

# **Python简单应用**

## 使用numpy进行数据分析


### 创建数组

numpy数组(Array)是一个值网格，所有类型都相同，并由非负整数元组索引。 
维数是数组的排名; 数组的形状是一个整数元组，给出了每个维度的数组大小。


```python
import numpy as np

# 向量
v = np.array([1, 2, 3, 4])
# 矩阵
m = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# 张量
t = np.array([
    [[1, 2, 3, 4], [5, 6, 7, 8]],
    [[9, 10, 11, 12], [13, 14, 15, 16]],
    [[17, 18, 19, 20], [21, 22, 23, 24]]
])
```


```python
print(v.shape)
print(m.shape)
print(t.shape)
```

numpy提供了许多创建数组的函数


```python
a = np.arange(0, 10, 1)
print(a)
a = np.linspace(0, 9, 10)
print(a)
O = np.zeros(3) # 零向量
print(O)
I = np.eye(3) # 单位矩阵
print(I)
```

### 切片

切片(Slicing): 与Python列表类似，可以对numpy数组进行切片。由于数组可能是多维的，因此必须为数组的每个维指定一个切片


```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print( a)
print( a[0:2 , 1:2] );
print( a[:, 1:2] )
print( a[:, 1])
```

### 数学运算
numpy内置的数学库可支持对数组元素进行多种数学运算


```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
b = np.array([[2, 3, 7], [1, 5, 2]])
print(b)
c = np.array([[2, -1], [-4, 6], [6, 5]])
print(c)
```


```python
# 基本代数运算
print(a ** 2)   # 对数组元素进行同种运算
print(a*c)  # 两个大小相同的数组，相同位置元素相乘
```


```python
# 线性代数运算
print(a.T) # 矩阵转置
print(np.dot(a, c)) # 矩阵乘法

m = np.array([[1,2], [4, 5]])
print(np.linalg.inv(m)) # 矩阵求逆
print(np.linalg.det(m)) # 行列式
```


```python
# 统计计算
data = np.random.random_sample(1000) * 100 # 生成1000个随机数集，默认0-1
print(data)

print(np.average(data)) #平均值
print(np.var(data)) #方差
print(np.sum(data)) # 求和
print(np.median(data)) # 中位数
```

## 使用matplotlib进行绘图

### 二维图像


```python
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

# 设置xy范围
plt.xlim(0, 20)
plt.ylim(-1.5, 1.5)

# 设置xy刻度
plt.xticks(np.arange(0, 21, 2))
plt.yticks(np.arange(-1.5, 1.6, 0.5))

# 设置label
plt.xlabel("x")
plt.ylabel("y")

# 设置标题
plt.title("sine")

x = np.array(np.arange(0, 20, 0.1))
y = np.sin(x * np.pi / 5)

# 画图
plt.plot(x, y, "r")

# 展示图
plt.show()
```

### 三维图像

用matplotlib画一个莫比乌斯环。


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')

u = np.linspace(0, 2*np.pi, endpoint=True, num=50)
v = np.linspace(-1, 1, endpoint=True, num=50)

u,v=np.meshgrid(u,v) #用meshgrid函数来产生三维绘图时的矩阵
print("shape of u:" + str(u.shape))
print("shape of v:" + str(v.shape))
print(u)
print(v)

u=u.flatten() #把u展开，变成一维数组
v=v.flatten() #把v展开，变成一维数组
x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
z = 0.5 * v * np.sin(u / 2.0)

tri = mtri.Triangulation(u, v)

ax.plot_trisurf(x, y, z, cmap="cool", triangles=tri.triangles)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1, 1)

plt.show()
```

### python学习资源

基础语法和应用开发：https://www.liaoxuefeng.com

numpy中文教程：https://www.numpy.org.cn/article/basics/understanding_numpy.html

matplotlib教程：https://www.runoob.com/w3cnote/matplotlib-tutorial.html


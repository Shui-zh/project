# Eigen 使用手册：C++ 线性代数库接口详解

> 适用版本：Eigen 3.4.x。文中带输出结果的核心代码片段，均已在 `g++ 13.3` + `-std=c++17` + Eigen 3.4.0 下实际编译运行验证。
> 官方站点：https://eigen.tuxfamily.org ｜ 文档：https://eigen.tuxfamily.org/dox/

---

## 目录

1. [快速上手](#1-快速上手)
2. [核心概念：Matrix 模板](#2-核心概念matrix-模板)
3. [构造与初始化](#3-构造与初始化)
4. [元素访问与遍历](#4-元素访问与遍历)
5. [块与切片](#5-块与切片)
6. [算术运算与别名问题](#6-算术运算与别名问题)
7. [归约、统计与广播](#7-归约统计与广播)
8. [逻辑、比较与条件选择](#8-逻辑比较与条件选择)
9. [Array 与逐元素数学](#9-array-与逐元素数学)
10. [形状与结构操作](#10-形状与结构操作)
11. [线性方程组求解](#11-线性方程组求解)
12. [矩阵分解详解](#12-矩阵分解详解)
13. [稀疏矩阵](#13-稀疏矩阵)
14. [几何模块](#14-几何模块)
15. [Map 与 Ref：外部内存与函数参数](#15-map-与-ref外部内存与函数参数)
16. [输入输出与调试](#16-输入输出与调试)
17. [性能实践](#17-性能实践)
18. [常见坑速查](#18-常见坑速查)
19. [API 速查表](#19-api-速查表)
20. [常见编译错误解读](#20-常见编译错误解读)

---

## 1. 快速上手

### 1.1 安装

Eigen 是 **header-only** 库，没有二进制依赖，只需头文件。

```bash
# Debian / Ubuntu
sudo apt install libeigen3-dev        # 头文件位于 /usr/include/eigen3

# macOS
brew install eigen

# 源码安装（推荐锁定 3.4.x）
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzf eigen-3.4.0.tar.gz -C /opt
```

编译时指定头文件路径：

```bash
g++ -std=c++17 -O2 -I/usr/include/eigen3 main.cpp -o main
```

### 1.2 CMake 集成

```cmake
cmake_minimum_required(VERSION 3.16)
project(demo CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)   # 提供 Eigen3::Eigen 目标

add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE Eigen3::Eigen)
target_compile_options(demo PRIVATE -O3 -march=native)
```

若系统未安装，可把 Eigen 作为子目录引入：`add_subdirectory(third_party/eigen)`。

### 1.3 最小示例

```cpp
#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;   m(1, 0) = 2.5;
    m(0, 1) = -1;  m(1, 1) = m(1, 0) + m(0, 1);
    std::cout << m << std::endl;
    return 0;
}
```

输出（注意：**Eigen 默认按列存储，(0,0) 是左上角，打印也按视觉布局**）：

```
  3  -1
2.5 1.5
```

### 1.4 头文件组织

| 头文件 | 内容 |
|---|---|
| `<Eigen/Core>` | 矩阵/数组类、基本运算、块操作、Map |
| `<Eigen/Dense>` | Core + LU + Cholesky + QR + SVD + 特征值 + Geometry（**最常用**） |
| `<Eigen/Geometry>` | 旋转、四元数、Transform |
| `<Eigen/Sparse>` | 稀疏矩阵与稀疏求解器入口 |
| `<Eigen/IterativeLinearSolvers>` | CG、BiCGSTAB 等迭代求解器 |
| `<Eigen/SparseCholesky>` | SimplicialLLT / SimplicialLDLT |
| `<Eigen/SparseLU>` `<Eigen/SparseQR>` | 稀疏 LU / QR |
| `<Eigen/Eigenvalues>` | 特征值分解 |
| `<Eigen/unsupported/...>` | 非官方模块（FFT、Kronecker 积、多项式求根等） |

> 只包含 `<Eigen/Dense>` 就能覆盖 90% 的稠密线性代数需求。

---

## 2. 核心概念：Matrix 模板

### 2.1 模板参数

```cpp
template<typename Scalar,
         int RowsAtCompileTime,
         int ColsAtCompileTime,
         int Options = 0,                          // RowMajor / ColMajor / AutoAlign / DontAlign
         int MaxRowsAtCompileTime = RowsAtCompileTime,
         int MaxColsAtCompileTime = ColsAtCompileTime>
class Matrix;
```

- `Scalar`：元素类型，`double`/`float`/`int`/`std::complex<double>` 等
- `RowsAtCompileTime` / `ColsAtCompileTime`：`Eigen::Dynamic`（即 `-1`）表示运行期确定
- `Options`：`Eigen::RowMajor` 或 `Eigen::ColMajor`（默认列主序），可叠加 `Eigen::DontAlign`
- `MaxRows/MaxCols`：动态尺寸矩阵的"上界"，用于栈上分配优化

### 2.2 常用类型别名

**矩阵**

| 别名 | 实际类型 |
|---|---|
| `MatrixXd` / `MatrixXf` | `Matrix<double/float, Dynamic, Dynamic>` |
| `MatrixXi` | `Matrix<int, Dynamic, Dynamic>` |
| `MatrixXcd` | `Matrix<complex<double>, Dynamic, Dynamic>` |
| `Matrix3d` / `Matrix3f` | 3×3 固定尺寸 |
| `Matrix2d` / `Matrix4d` | 2×2 / 4×4 固定尺寸 |
| `Matrix<double, 3, Dynamic>` | 3 行、列数动态 |

**向量**（本质是只有一列的矩阵）

| 别名 | 实际类型 |
|---|---|
| `VectorXd` | `Matrix<double, Dynamic, 1>` |
| `Vector3d` / `Vector2f` | `Matrix<double, 3, 1>` / `Matrix<float, 2, 1>` |
| `RowVectorXd` | `Matrix<double, 1, Dynamic>` |
| `Vector3d::UnitX()` | `(1,0,0)`（静态方法返回单位向量） |

**数组**（逐元素运算）

| 别名 | 实际类型 |
|---|---|
| `ArrayXd` | `Array<double, Dynamic, 1>` |
| `ArrayXXd` | `Array<double, Dynamic, Dynamic>` |
| `Array3d` | 固定 3 元素 |

**稀疏**

| 别名 | 实际类型 |
|---|---|
| `SparseMatrix<double>` | 列主序压缩稀疏矩阵（默认） |
| `SparseMatrix<double, RowMajor>` | 行主序 |
| `SparseVector<double>` | 稀疏向量 |

### 2.3 固定尺寸 vs 动态尺寸

| | 固定尺寸（`Matrix3d`） | 动态尺寸（`MatrixXd`） |
|---|---|---|
| 尺寸 | 编译期已知 | 运行期 `resize` |
| 存储 | 栈上（成员变量内） | 堆上 |
| 速度 | 快，可完全展开、向量化 | 稍慢，但可控 |
| 适用 | 尺寸小且固定（≤ 4~16 维，几何/机器人） | 大矩阵、尺寸运行时才知 |

```cpp
Eigen::Matrix3d A;        // 无需指定尺寸
Eigen::MatrixXd B(3, 3);  // 必须指定尺寸
// Eigen::MatrixXd C;     // 空矩阵，0×0，直接 A = C 会运行期断言失败
```

> **经验法则**：4×4 以下用固定尺寸；超过 32×32 一定用动态；介于两者视情况（栈空间限制见 `EIGEN_STACK_ALLOCATION_LIMIT`，默认 128KB）。

### 2.4 存储顺序

```cpp
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> R(2, 3);
```

- 默认 **列主序（Column-major）**，与 Fortran/BLAS/LAPACK 一致
- 行主序在按行遍历、与 C 数组/图像缓冲区对接时更快
- `data()` 返回的裸指针顺序由存储顺序决定

### 2.5 表达式模板（Expression Templates）

Eigen 的运算**不立即求值**，而是构造表达式树，在赋值或 `.eval()` 时才计算。

```cpp
Eigen::MatrixXd A, B, C;
// 下面一行只做一次循环、一个临时变量都没有：
C = A + B + A * B;
```

好处是零开销抽象；代价是：

1. **`auto` 陷阱**：`auto` 会捕获表达式而非结果，右侧临时对象析构后悬垂。
2. **别名（aliasing）问题**：见 [6.4](#64-aliasing别名问题)。
3. 模板报错极长：见 [第 20 节](#20-常见编译错误解读)。

```cpp
// 错误：expr 是对临时对象的引用，A+B 结束后即销毁
auto expr = A + B;
// 正确
Eigen::MatrixXd expr = A + B;
// 或者显式求值
auto expr = (A + B).eval();
```

---

## 3. 构造与初始化

### 3.1 构造方式

```cpp
#include <Eigen/Dense>
using namespace Eigen;

MatrixXd m1(3, 4);                       // 3x4，未初始化
MatrixXd m2(3, 4); m2.setZero();         // 3x4 全 0
Matrix3d m3;                             // 3x3 未初始化
VectorXd v1(5);                          // 长度 5
Vector3d v2(1.0, 2.0, 3.0);              // 固定尺寸向量可直接给元素
Vector3d v3(v2);                         // 拷贝
MatrixXd m4 = MatrixXd::Zero(2, 2);      // 从表达式初始化
MatrixXd m5 = m3;                        // 固定 -> 动态，OK
// Matrix3d m6 = m4;                     // 动态 -> 固定：编译期报 YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES
```

> **极易踩的坑**：`VectorXd v(1, 2);` 想初始化两个元素是**编译错误**（动态向量没有 (a,b) 构造，只有 (size)）。正确写法：
> ```cpp
> VectorXd v(2); v << 1, 2;
> ```
> 同理 `VectorXd b(1,2,3,4)` 会报 `THIS_METHOD_IS_ONLY_FOR_VECTORS_OF_A_SPECIFIC_SIZE`。

### 3.2 逗号初始化（按行填充！）

```cpp
Matrix3d A;
A << 1, 2, 3,
     4, 5, 6,
     7, 8, 10;     // 数值按“行”顺序填入，与存储顺序无关
```

输出：

```
 1  2  3
 4  5  6
 7  8 10
```

```cpp
RowVectorXd rv(3);
rv << 1, 2, 3;

VectorXd v(4);
v << 1, 2, 3, 4;

// 也可以追加：已初始化的矩阵末尾继续 <<
MatrixXd M(2, 2);
M << 1, 2, 3, 4;
M << 5, 6, 7, 8;    // 覆盖，不追加
```

> 逗号初始化不能跨越已存在的元素个数，超出会触发 `eigen_assert`。首次 `<<` 前矩阵尺寸必须正确。

### 3.3 特殊矩阵生成器（静态方法）

| 表达式 | 说明 |
|---|---|
| `MatrixXd::Zero(r, c)` | 全 0 |
| `MatrixXd::Ones(r, c)` | 全 1 |
| `MatrixXd::Constant(r, c, v)` | 全 v |
| `MatrixXd::Identity(r, c)` | 单位阵（非方阵时左上角为单位） |
| `MatrixXd::Random(r, c)` | 均匀分布 `[-1, 1]` |
| `VectorXd::LinSpaced(n, lo, hi)` | n 个等距点，**含端点** |
| `VectorXd::LinSpaced(n, lo, hi)` 的 `Sequential_t` 版本 | 3.4 已废弃，勿用 |
| `MatrixXd::NullaryExpr(r, c, functor)` | 由函数生成 |

```cpp
std::cout << MatrixXd::Identity(3, 3) << "\n";
std::cout << MatrixXd::Constant(2, 3, 7.5) << "\n";
std::cout << VectorXd::LinSpaced(5, 0.0, 1.0).transpose() << "\n";
// 0 0.25 0.5 0.75 1
```

> `MatrixXd::Random` 是 `[-1,1]` 均匀分布，**不是**正态分布。正态分布需自己用 `std::normal_distribution` + `NullaryExpr`。

### 3.4 原地设置（已分配的矩阵）

```cpp
MatrixXd m(3, 3);
m.setZero();
m.setOnes();
m.setIdentity();
m.setConstant(2.5);
m.setRandom();
m.fill(7.0);          // 同 setConstant
m.diagonal().setConstant(1.0);
```

### 3.5 尺寸调整

```cpp
MatrixXd m(2, 3);
m.resize(4, 5);              // 旧内容丢失，不初始化
m.setZero();                 // 记得初始化

VectorXd v(3); v << 1, 2, 3;
v.conservativeResize(5);     // 保留已有元素，新增部分不初始化
v.tail(2) << 4, 5;
// v = 1 2 3 4 5

v.conservativeResize(2);     // 截断，保留前 2 个
```

`conservativeResize` 支持 `(rows, cols)` 与带行列保留数量的三参数版本：

```cpp
m.conservativeResize(4, 4);           // 保留左上角重叠部分
```

### 3.6 从已有数据构造

```cpp
double raw[6] = {1, 2, 3, 4, 5, 6};
// 推荐用 Map 零拷贝（见第 15 节）
Eigen::Map<Eigen::Matrix<double, 2, 3>> m(raw);

// 小数据也可逐个赋值或用逗号
std::vector<double> vec = {1, 2, 3};
Eigen::Map<Eigen::VectorXd> v(vec.data(), vec.size());
```

---

## 4. 元素访问与遍历

### 4.1 访问方式

```cpp
MatrixXd m(2, 3);
m(0, 1) = 5.0;              // (row, col)，最常用；debug 下有越界断言
double x = m(1, 2);
m.coeffRef(0, 0) += 1.0;    // 无断言版本，热循环里略快
double y = m.coeff(0, 0);   // 只读，无断言

Vector3d v(1, 2, 3);
v(0); v[1]; v.x(); v.y(); v.z(); v.w();   // w() 仅 4 维向量
v(1) = 9;                   // 向量可用单索引

double* p = m.data();       // 裸指针，按存储顺序（默认列主序）
```

> 对矩阵也可以用单索引 `m(i)`，等价于按存储顺序的线性访问（列主序下 `i = row + col*rows`）。仅在明确知道存储顺序时使用。

### 4.2 遍历

```cpp
MatrixXd m = MatrixXd::Random(3, 4);

// 1) 双重循环
for (Eigen::Index i = 0; i < m.rows(); ++i)
    for (Eigen::Index j = 0; j < m.cols(); ++j)
        m(i, j) *= 2;

// 2) 按列遍历（列主序下缓存友好）
for (Eigen::Index j = 0; j < m.cols(); ++j)
    for (Eigen::Index i = 0; i < m.rows(); ++i)
        m(i, j) += 1;

// 3) 范围 for（Eigen 3.4+，按存储顺序）
for (double& x : m) x = 0.0;
for (auto x : m) std::cout << x << " ";

// 4) 逐元素函数（推荐，可向量化）
m = m.unaryExpr([](double t) { return t > 0 ? t : 0.0; });
```

> **不要**在遍历时改变矩阵尺寸（`resize`），会导致迭代器/指针失效。

### 4.3 索引类型

`Eigen::Index` 是 `std::ptrdiff_t` 的别名；与 `int` 混用会有符号比较警告，建议统一用 `Eigen::Index`。

```cpp
Eigen::Index r, c;
double mx = m.maxCoeff(&r, &c);   // 同时返回最大值的行列位置
```

---

## 5. 块与切片

Eigen 的切片通过"块（block）"实现，返回的是**视图**，赋值即写回原矩阵。

### 5.1 常用块接口

| 表达式 | 含义 |
|---|---|
| `m.block(i, j, p, q)` | 从 `(i,j)` 起 `p×q` 的子块 |
| `m.block(i, j)` | 从 `(i,j)` 到右下角 |
| `m.topLeftCorner(p, q)` / `topLeftCorner(p)` | 左上角 |
| `m.topRightCorner(p, q)` / `bottomLeftCorner` / `bottomRightCorner` | 其余三角 |
| `m.row(i)` / `m.col(j)` | 第 i 行 / 第 j 列 |
| `m.topRows(n)` / `m.bottomRows(n)` | 首/尾 n 行 |
| `m.leftCols(n)` / `m.rightCols(n)` | 首/尾 n 列 |
| `m.middleRows(i, n)` / `m.middleCols(j, n)` | 中间 n 行/列 |
| `v.head(n)` / `v.tail(n)` | 向量首/尾 n 个元素 |
| `v.segment(i, n)` | 从 i 起 n 个元素 |
| `v.segment<n>(i)` | 模板版本，长度编译期已知，更快 |
| `m.block<p, q>(i, j)` | 模板版本，尺寸编译期已知 |

### 5.2 示例

```cpp
MatrixXd B = MatrixXd::Random(4, 4);

std::cout << B.topLeftCorner(2, 2) << "\n";
std::cout << B.row(1) << "\n";                     // 1x4 行视图
std::cout << B.col(2).head(2).transpose() << "\n";
std::cout << B.block(1, 1, 2, 3) << "\n";          // 2x3

VectorXd long_vec = VectorXd::LinSpaced(10, 0, 9);
std::cout << long_vec.segment(2, 4).transpose() << "\n";   // 2 3 4 5
std::cout << long_vec.tail(3).transpose() << "\n";         // 7 8 9
```

写入块：

```cpp
MatrixXd C = MatrixXd::Zero(4, 4);
C.block(0, 0, 2, 2) = Matrix2d::Ones();   // 左上角 2x2 置 1
C(3, 3) = 9;
C.row(0) *= 2;
C.col(1).setZero();
C.bottomRightCorner(2, 2).diagonal().setConstant(5);
```

### 5.3 隔行/隔列取样

Eigen 3.4 **没有** NumPy 那样的 `M[::2, ::2]` 步长切片，也没有 stride 版 `block`。需要隔行/隔列取样时，只能：

```cpp
MatrixXd M = MatrixXd::Random(6, 6);

// 方法 1：索引循环拷贝出子矩阵（最直观）
MatrixXd sub(3, 3);
for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
        sub(i, j) = M(2 * i, 2 * j);

// 方法 2：整列/整行抽取（块只能取连续区间，跨步需要逐列赋值）
MatrixXd oddCols(6, 3);
for (int j = 0; j < 3; ++j) oddCols.col(j) = M.col(2 * j);
```

对**连续内存 + 固定步长**的场景（如图像缓冲区），用 `Map` + `Stride` 做零拷贝视图，见 15.3 节。

---

## 6. 算术运算与别名问题

### 6.1 基本运算

| 表达式 | 含义 |
|---|---|
| `A + B`, `A - B` | 逐元素加减（尺寸必须相同） |
| `A * B` | **矩阵乘法** |
| `A * s`, `s * A`, `A / s` | 标量乘除 |
| `A.cwiseProduct(B)` | 逐元素乘（Hadamard 积） |
| `A.cwiseQuotient(B)` | 逐元素除 |
| `A.array() * B.array()` | 同上，Array 语义 |
| `-A` | 取负 |
| `A.transpose()` | 转置（视图） |
| `A.adjoint()` | 共轭转置（实矩阵 = 转置） |
| `A.conjugate()` | 逐元素共轭 |
| `A.dot(v)` / `v1.dot(v2)` | 点积（内积） |
| `v1.cross(v2)` | 叉积（仅 3 维） |
| `v1.outer(v2)` | 外积（返回矩阵） |
| `A *= 2`, `A += B` | 复合赋值，原地 |

```cpp
MatrixXd A = MatrixXd::Random(3, 3);
MatrixXd B = MatrixXd::Random(3, 3);
VectorXd x = VectorXd::LinSpaced(3, 1, 3);

A * B;                    // 矩阵乘
A.cwiseProduct(B);        // 逐元素
A * x;                    // 矩阵-向量
x.transpose() * A * x;    // 二次型，结果是 1x1 矩阵
x.dot(x);                 // 标量
A.trace();                // 迹
```

> `x.transpose() * A * x` 的结果类型是 1×1 矩阵而非 `double`，需要 `.value()` 或 `(x.transpose()*A*x)(0,0)` 取值。

### 6.2 维度检查

矩阵乘法的维度在**编译期**（固定尺寸）或**运行期断言**（动态尺寸）检查：

```cpp
MatrixXd A(3, 4), B(3, 4);
// A * B;   // 运行期断言：invalid sizes
MatrixXd C = A * B.transpose();   // 3x4 * 4x3 = 3x3，OK
```

### 6.3 原地运算与临时对象

```cpp
MatrixXd M = MatrixXd::Constant(2, 2, 2.0);
M *= 3.0;
M += MatrixXd::Ones(2, 2);
M = M * M;              // 安全：Eigen 会自动引入临时变量（矩阵乘有特殊处理）
```

`noalias()` 用于承诺"左侧不与右侧重叠"，避免临时变量：

```cpp
MatrixXd C(n, n);
C.noalias() = A * B;    // 直接写入 C，省一次拷贝
C.noalias() += A * B;
```

> 仅在**确定无别名**时使用 `noalias()`。`C = C * A` 这类自引用运算加 `noalias()` 会得到错误结果。

### 6.4 aliasing（别名）问题

当左侧出现在右侧时，Eigen 默认会检测并自动插入临时变量 —— 但**转置、块赋值等少数情形无法自动处理**，会直接触发断言。

```cpp
Matrix2d T;
T << 1, 2, 3, 4;

// 错误：运行期断言 "aliasing detected during transposition"
// T = T.transpose();

// 正确做法 1：原地转置（仅方阵）
T.transposeInPlace();

// 正确做法 2：显式求值
T = T.transpose().eval();

// 正确做法 3：赋值给另一个变量（非别名场景）
Matrix2d U = T.transpose();
```

其他典型别名场景：

```cpp
// 块赋值重叠
// m.topLeftCorner(2,2) = m.bottomRightCorner(2,2);   // 危险，元素重叠

// 安全写法
MatrixXd tmp = m.bottomRightCorner(2, 2);
m.topLeftCorner(2, 2) = tmp;

// 逐元素自乘：安全（Eigen 自动处理）
m = m.array() * m.array();
```

### 6.5 标量类型混用

```cpp
MatrixXd Ad = MatrixXd::Random(3, 3);
MatrixXf Af = Ad.cast<float>();       // 显式转换，推荐
VectorXi vi = VectorXd::LinSpaced(3, 1, 3).cast<int>();   // 1 2 3
// MatrixXd bad = Ad + Af;            // 编译期报错 YOU_MIXED_DIFFERENT_NUMERIC_TYPES
```

---
## 7. 归约、统计与广播

### 7.1 全矩阵归约

| 方法 | 结果 |
|---|---|
| `A.sum()` | 所有元素和 |
| `A.prod()` | 所有元素积 |
| `A.mean()` | 平均值 |
| `A.minCoeff()` / `A.maxCoeff()` | 最小值 / 最大值 |
| `A.minCoeff(&i, &j)` / `A.maxCoeff(&i, &j)` | 同时返回位置 |
| `A.trace()` | 迹（对角线和） |
| `A.norm()` | Frobenius 范数（√元素平方和） |
| `A.squaredNorm()` | 元素平方和 |
| `A.lpNorm<1>()` | 1-范数（元素绝对值之和） |
| `A.lpNorm<Eigen::Infinity>()` | 无穷范数（最大绝对值） |
| `A.all()` / `A.any()` | 所有/任一元素非零 |
| `A.count()` | 非零元素个数 |
| `A.diagonal().sum()` | 迹（等价写法） |

```cpp
MatrixXd R(2, 3);
R << 1, 2, 3,
     4, 5, 6;

R.sum();           // 21
R.mean();          // 3.5
R.prod();          // 720
R.minCoeff();      // 1
R.maxCoeff();      // 6
R.norm();          // 9.53939
R.squaredNorm();   // 91
R.lpNorm<1>();     // 21
R.topLeftCorner(2, 2).trace();   // 6
```

> `norm()` 对向量是欧氏长度，对矩阵是 Frobenius 范数（**不是**谱范数）。谱范数用 `JacobiSVD` 的最大奇异值。

### 7.2 按行/按列归约（partial reduction）

```cpp
MatrixXd R(2, 3);
R << 1, 2, 3,
     4, 5, 6;

R.colwise().sum();        // 行向量 [5 7 9]      —— 每列求和
R.rowwise().sum();        // 列向量 [6 15]       —— 每行求和
R.colwise().mean();       // 行向量 [2.5 3.5 4.5]
R.rowwise().mean();       // 列向量 [2 5]
R.colwise().maxCoeff();   // [4 5 6]
R.colwise().norm();       // [4.123 5.385 6.708]
R.rowwise().squaredNorm();
R.colwise().count();      // 每列非零个数
R.colwise().all();        // 每列是否全非零 → 行向量
```

返回值形状规则：

| 表达式 | 形状 |
|---|---|
| `A.colwise().xxx()` | `1 × cols` **行向量** |
| `A.rowwise().xxx()` | `rows × 1` **列向量** |

### 7.3 广播（broadcasting）

`rowwise()` / `colwise()` 配合 `+ - * /` 可以对每行/每列施加同一个向量。**规则非常反直觉，务必记牢**：

| 表达式 | 右侧 v 的形状 | 语义 |
|---|---|---|
| `M.rowwise() op v` | **1×N 行向量** | 把 v 复制到每一行，`M(i,j) op v(0,j)` |
| `M.colwise() op v` | **N×1 列向量** | 把 v 复制到每一列，`M(i,j) op v(i,0)` |

实测（`M` 为 3×3，值为 10..90）：

```cpp
Matrix3d M;
M << 10, 20, 30,
     40, 50, 60,
     70, 80, 90;

RowVector3d rv(1, 2, 3);
Vector3d    cv(1, 2, 3);

M.rowwise() - rv;    // 每行减 [1 2 3]：
                     //  9  18  27
                     // 39  48  57
                     // 69  78  87

M.colwise() - cv;    // 每列减 [1 2 3]^T：
                     //  9  19  29
                     // 38  48  58
                     // 67  77  87
```

以下两种写法会**编译失败**（静态断言）：

```cpp
// M.rowwise() - cv;   // YOU_PASSED_A_COLUMN_VECTOR_BUT_A_ROW_VECTOR_WAS_EXPECTED
// M.colwise() - rv;   // YOU_PASSED_A_ROW_VECTOR_BUT_A_COLUMN_VECTOR_WAS_EXPECTED
```

### 7.4 常见广播任务

```cpp
MatrixXd D = MatrixXd::Random(3, 3);

// 1) 每行是一个样本（N×D 布局）时：每列减去该列均值（即按特征中心化，PCA 常用）
MatrixXd C1 = D.rowwise() - D.colwise().mean();

// 2) 每行减去该行均值
//    注意：D.rowwise() - D.rowwise().mean() 编译不过（右侧是列向量），
//    必须先把列向量 replicate 成行广播矩阵
MatrixXd C2 = D - D.rowwise().mean().replicate(1, D.cols());

// 3) 每列除以该列范数（归一化）
MatrixXd Z = C1.array().rowwise() / C1.colwise().norm().array();

// 4) 每列减去该列最小值
MatrixXd S = D.rowwise() - D.colwise().minCoeff();

// 5) 每行加上同一个 1×N 行向量 rv
//    D.rowwise() += rv;

// 6) 每列乘以同一个 N×1 列向量 cv（逐元素）
//    D.colwise() *= cv;
```

> 口诀：**`rowwise` 配行向量（1×N），`colwise` 配列向量（N×1）**。这跟"按行操作该用列向量"的直觉相反，因为 Eigen 的语义是"把这个向量按行/按列复制铺满"，而不是"每行一个标量"。

---

## 8. 逻辑、比较与条件选择

### 8.1 矩阵比较

```cpp
MatrixXd A = MatrixXd::Random(3, 3);
MatrixXd B = A;

A == B;                    // 逐元素精确相等，返回 bool（B 是 A 的拷贝，故为 true）；浮点结果不宜这样比较
A.isApprox(B);             // 推荐：范数意义下近似相等
A.isApprox(B, 1e-9);       // 指定精度
A.isUnitary();             // 是否正交/酉（R^T R = I）
A.isIdentity();            // 是否单位阵
A.isZero();
A.isConstant(1.0);
A.hasNaN();                // 含 NaN
A.allFinite();             // 全部有限（无 NaN/Inf）
```

### 8.2 逐元素比较（必须转 Array）

```cpp
MatrixXd P(2, 2);
P << 1, -2,
     3, -4;

(P.array() > 0);                    // 布尔表达式
(P.array() > 0).count();            // 2
(P.array() > 0).all();              // false
(P.array() > 0).any();              // true

// 条件选择：cond.select(then, else)
(P.array() > 0).select(P, 0.0);     // 负数置 0（ReLU）
(P.array() > 0).select(P, -P);      // 取绝对值

// 逐元素 min/max
P.cwiseMin(0.0);                    // 下截断
P.cwiseMax(0.0);                    // 上截断
P.cwiseAbs();                       // 绝对值
P.cwiseAbs2();                      // 平方
```

输出：

```
1 0
3 0
```

### 8.3 查找与掩码

```cpp
MatrixXd M = MatrixXd::Random(3, 3);
Eigen::Index r, c;
M.maxCoeff(&r, &c);
M.minCoeff(&r, &c);

// 统计满足条件的元素个数矩阵（按列）
(M.array() > 0).colwise().count();

// 用 all()/any() 做整体判断
if ((M.array() < 0).any()) { /* ... */ }
```

---

## 9. Array 与逐元素数学

### 9.1 Matrix 与 Array 的区别

- `Matrix`：`*` 是矩阵乘法，`+ -` 逐元素
- `Array`：**所有运算符都是逐元素**，`*` 就是 Hadamard 积，还支持 `sin/cos/exp` 等数学函数

```cpp
ArrayXd a(3), b(3);
a << 1, 2, 3;
b << 4, 5, 6;
a * b;            // 逐元素 [4 10 18]
a + b;
a.sin();
```

### 9.2 互相转换

```cpp
MatrixXd M = MatrixXd::Random(3, 3);
ArrayXXd A = M.array();          // Matrix -> Array（视图）
MatrixXd M2 = A.matrix();        // Array -> Matrix（视图）
MatrixXd M3 = M.array().square().matrix();   // 链式转换
```

### 9.3 逐元素数学函数

| 类别 | 函数 |
|---|---|
| 三角 | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| 双曲 | `sinh`, `cosh`, `tanh` |
| 指数对数 | `exp`, `log`, `log10`, `log1p`, `expm1` |
| 幂与根 | `sqrt`, `rsqrt`, `pow(p)`, `square`, `cube`, `inverse`（逐元素倒数） |
| 取整 | `floor`, `ceil`, `round`, `rint` |
| 符号与绝对值 | `abs`, `abs2`, `sign`, `cwiseAbs` |
| 其他 | `pow`, `binaryExpr`, `unaryExpr`, `isNaN`, `isFinite` |

```cpp
ArrayXd a = ArrayXd::LinSpaced(5, 0.0, 1.0);

a.sin();          // 0 0.247404 0.479426 0.681639 0.841471
a.exp();          // 1 1.28403 1.64872 2.117 2.71828
a.sqrt();         // 0 0.5 0.707107 0.866025 1
a.pow(2);         // 0 0.0625 0.25 0.5625 1
(a - 0.5).abs();  // 0.5 0.25 0 0.25 0.5
(a * 3).round();  // 0 1 2 2 3
(a * 3).floor();  // 0 0 1 2 3
(a * 3).ceil();   // 0 1 2 3 3
```

### 9.4 自定义逐元素运算

```cpp
// 一元
MatrixXd E = MatrixXd::Random(2, 3);
MatrixXd relu = E.unaryExpr([](double t) { return t > 0 ? t : 0.0; });

// 二元
ArrayXd x = ArrayXd::LinSpaced(5, 0.0, 1.0);
ArrayXd y = x.binaryExpr(x, [](double u, double v) { return std::atan2(u, v); });

// 带状态的 functor（注意 functor 会被拷贝，用引用捕获需小心）
struct Scale { double k; double operator()(double t) const { return k * t; } };
MatrixXd scaled = E.unaryExpr(Scale{2.0});
```

### 9.5 复数

```cpp
// Eigen 已内置 MatrixXcd = Matrix<complex<double>, Dynamic, Dynamic>
MatrixXcd C(2, 2);
C << complex<double>(1, 0), complex<double>(0, 1),
     complex<double>(2, -1), complex<double>(3, 0);

C.real();          // 实部
C.imag();          // 虚部
C.conjugate();     // 逐元素共轭
C.adjoint();       // 共轭转置
C.transpose();     // 仅转置
C.cwiseAbs();      // 模
C.norm();
```

---

## 10. 形状与结构操作

### 10.1 转置、共轭、重塑

| 表达式 | 说明 | 是否拷贝 |
|---|---|---|
| `A.transpose()` | 转置 | 视图 |
| `A.adjoint()` | 共轭转置 | 视图 |
| `A.conjugate()` | 逐元素共轭 | 视图 |
| `A.transposeInPlace()` | 原地转置（**仅方阵**） | 原地 |
| `A.adjointInPlace()` | 原地共轭转置（仅方阵） | 原地 |
| `A.reshaped(r, c)` | 重塑为 r×c，默认按列展开 | 视图（3.4+） |
| `A.reshaped<RowMajor>(r, c)` | 按行展开重塑 | 视图 |
| `A.reverse()` | 反转元素（列主序顺序） | 视图 |
| `A.rowwise().reverse()` | 每行内反转 | 视图 |
| `A.colwise().reverse()` | 每列内反转 | 视图 |
| `A.replicate(m, n)` | 行列各复制 m、n 次 | 表达式 |
| `A.diagonal()` / `A.diagonal(k)` | 主/第 k 条对角线 | 视图 |
| `v.asDiagonal()` | 向量 → 对角矩阵 | 包装 |
| `A.asDiagonal().toDenseMatrix()` | 转成真实矩阵 | 拷贝 |

```cpp
MatrixXd Q(2, 3);
Q << 1, 2, 3,
     4, 5, 6;

Q.reshaped(3, 2);              // 按列展开：1 5 / 4 3 / 2 6
Q.reshaped<RowMajor>(3, 2);    // 按行展开：1 2 / 3 4 / 5 6

VectorXi::LinSpaced(4, 1, 4).reverse();        // 4 3 2 1
Vector2d(1, 2).replicate(2, 1).transpose();    // 1 2 1 2

Matrix4d A = Matrix4d::Random();
A.diagonal(1).transpose();      // 超对角线（3 个元素）
A.diagonal() = Vector4d(1, 2, 3, 4);   // 写对角线

Vector3d dv(1, 2, 3);
dv.asDiagonal().toDenseMatrix();   // 3x3 对角阵
```

> `reshaped` 是 Eigen 3.4 新增；3.3 及更早需要用 `Map` 或手动循环。

### 10.2 拼接

Eigen 没有 `hstack`/`vstack`，用三种方式：

```cpp
// 1) 逗号初始化（尺寸必须预先正确）
MatrixXd H(2, 4);
H << MatrixXd::Ones(2, 2), MatrixXd::Zero(2, 2);

MatrixXd V(4, 2);
V << MatrixXd::Ones(2, 2), MatrixXd::Zero(2, 2);

// 2) 逐块赋值
MatrixXd K = MatrixXd::Zero(3, 6);
K.block(0, 0, 3, 3) = MatrixXd::Identity(3, 3);
K.block(0, 3, 3, 3) = MatrixXd::Constant(3, 3, 2.0);

// 3) 单位阵左右扩列（解方程时常用）
MatrixXd Aug(3, 4);
Aug << A, b;                  // A 为 3x3，b 为 3x1
```

> 逗号初始化里放表达式时，**逗号会被解析为逗号运算符导致意外**（尤其涉及 `,` 分隔的函数调用）。必要时给表达式加括号：`H << (A + B), C;`。

### 10.3 三角视图与自伴视图

只读写矩阵的一半，省一半计算：

```cpp
MatrixXd A = MatrixXd::Random(4, 4);

// 三角视图（Lower / Upper / UnitLower / UnitUpper / StrictlyLower ...）
A.triangularView<Lower>();                 // 视图
MatrixXd L = A.triangularView<Lower>();    // 拷贝成下三角矩阵
A.triangularView<Lower>().transpose();

// 三角求解
VectorXd b = VectorXd::Ones(4);
MatrixXd Lm = MatrixXd::Random(4, 4).triangularView<Lower>();
Lm.diagonal().array() += 3;                // 保证对角占优
VectorXd y = Lm.triangularView<Lower>().solve(b);
VectorXd z = Lm.triangularView<Lower>().transpose().solve(b);

// 自伴视图：只用下（或上）三角，把矩阵当作对称阵
Matrix3d S;
S << 4, 1, 1,
     1, 3, 0,
     1, 0, 2;
S.selfadjointView<Lower>() * b.head(3);            // 自伴矩阵乘向量
S.selfadjointView<Lower>().ldlt().solve(b.head(3)); // 自伴求解
```

> `triangularView` / `selfadjointView` 不能直接 `std::cout <<`，需要赋给矩阵或调用 `.toDenseMatrix()`。

---
## 11. 线性方程组求解

### 11.1 选择分解方法的决策表

解 `A x = b`，按矩阵性质选：

| 矩阵性质 | 推荐分解 | 复杂度 | 特点 |
|---|---|---|---|
| 一般方阵、可逆 | `partialPivLu()` | O(n³) | 最快，最常用 |
| 方阵、需判定可逆/求秩/求核 | `fullPivLu()` | 稍慢 | 数值最稳，能处理奇异 |
| 对称正定 (SPD) | `llt()` | O(n³/3) | 最快，但仅 SPD |
| 对称半正定/不定 | `ldlt()` | O(n³/3) | 比 LLT 宽容，不要求正定 |
| 长方形、满秩最小二乘 | `householderQr()` | O(mn²) | 稳定，不选主元 |
| 秩亏、需知秩 | `colPivHouseholderQr()` | 稍慢 | 列主元，可求秩/核 |
| 严重秩亏、要最小范数解 | `completeOrthogonalDecomposition()` | 慢 | 能算伪逆 |
| 病态、要奇异值/条件数 | `jacobiSvd()` / `bdcSvd()` | 最慢 | 最稳健 |
| 大规模稀疏 SPD | `SimplicialLLT` / `ConjugateGradient` | 依稀疏度 | 见第 13 节 |

### 11.2 基本用法

```cpp
Matrix3d A;
A << 4, -2, 1,
     -2, 4, -2,
     1, -2, 4;
Vector3d b(11, -16, 17);

// 方式 1：一步求解（内部会构造分解对象）
Vector3d x1 = A.partialPivLu().solve(b);   // 1 -2 3
Vector3d x2 = A.fullPivLu().solve(b);
Vector3d x3 = A.colPivHouseholderQr().solve(b);

// 方式 2：复用分解对象（推荐：多次求解同一矩阵时）
PartialPivLU<Matrix3d> lu(A);
Vector3d y1 = lu.solve(b);
Vector3d y2 = lu.solve(b * 2);             // 分解只做一次

std::cout << "残差 = " << (A * x1 - b).norm() << "\n";   // 0
```

SPD 矩阵：

```cpp
Matrix3d S;
S << 4, 1, 1,
     1, 3, 0,
     1, 0, 2;
Vector3d bs(1, 2, 3);

Vector3d xs1 = S.llt().solve(bs);    // -0.368421 0.789474 1.68421
Vector3d xs2 = S.ldlt().solve(bs);   // 同上

LLT<Matrix3d> llt(S);
std::cout << llt.info() << "\n";                  // 0 表示 Success
std::cout << llt.matrixL().toDenseMatrix() << "\n";  // L（下三角）
std::cout << llt.rcond() << "\n";                 // 倒数条件数估计
```

### 11.3 多右端项

`solve()` 的右端可以是矩阵，一次解多个方程组：

```cpp
Matrix3d B = Matrix3d::Identity();
Matrix3d X = A.partialPivLu().solve(B);    // 等价于求 A 的逆
```

### 11.4 超定 / 欠定（最小二乘）

```cpp
// 超定：4 个方程 3 个未知数
MatrixXd A2(4, 3);
A2 << 1, 2, 3,
      4, 5, 6,
      7, 8, 10,
      1, 1, 1;
VectorXd b2(4);
b2 << 1, 2, 3, 4;

VectorXd x = A2.colPivHouseholderQr().solve(b2);   // 最小二乘解
std::cout << "|Ax-b| = " << (A2 * x - b2).norm() << "\n";   // 3.31662

// 欠定（方程少于未知数）：主张量 QR 给出最小范数解
MatrixXd R1(2, 3);
R1 << 1, 2, 3,
      2, 4, 7;
VectorXd b3(2);
b3 << 1, 2;
VectorXd xmin = R1.completeOrthogonalDecomposition().solve(b3);
```

### 11.5 伪逆（Moore-Penrose）

```cpp
MatrixXd R1(2, 3);
R1 << 1, 2, 3,
      2, 4, 7;

// 方法 1：完全正交分解（推荐，快且稳）
MatrixXd pinv = R1.completeOrthogonalDecomposition().pseudoInverse();

// 方法 2：SVD 手写（可控截断阈值）
JacobiSVD<MatrixXd> svd(R1, ComputeThinU | ComputeThinV);
double tol = 1e-12 * std::max(R1.rows(), R1.cols()) * svd.singularValues()(0);
VectorXd invS = svd.singularValues().array().inverse();   // 需要先把小奇异值清零
for (int i = 0; i < invS.size(); ++i) invS(i) = svd.singularValues()(i) > tol ? invS(i) : 0.0;
MatrixXd pinv2 = svd.matrixV() * invS.asDiagonal() * svd.matrixU().transpose();

std::cout << (R1 * pinv * R1 - R1).norm() << "\n";   // ≈ 0
```

> `JacobiSVD` **没有** `inverse()` 成员；求逆必须像上面这样手动拼，或改用 `A.inverse()`（仅适用方阵可逆情形）。

### 11.6 求解质量检查

永远检查残差，而不是盲信 `solve`：

```cpp
VectorXd x = A.colPivHouseholderQr().solve(b);
double res = (A * x - b).norm();
double rel = res / b.norm();
if (rel > 1e-8) { /* 病态或秩亏，改用 SVD/正则化 */ }

// 更严格的 backwards error
double berr = (A * x - b).norm() / (A.norm() * x.norm() + b.norm());
```

### 11.7 求逆与行列式

```cpp
Matrix3d A = Matrix3d::Random();
Matrix3d Ainv = A.inverse();                 // 内部用 PartialPivLU
double det = A.determinant();                // 内部用 LU
bool inv = A.fullPivLu().isInvertible();

// 更好：避免显式求逆，直接解
Vector3d b(1, 2, 3);
Vector3d x = A.partialPivLu().solve(b);      // 比 A.inverse() * b 快且更准
```

> **不要**写 `x = A.inverse() * b`。除非确实需要逆矩阵本身。

---

## 12. 矩阵分解详解

### 12.1 LU 分解

```cpp
#include <Eigen/LU>   // <Eigen/Dense> 已包含

Matrix3d A;
A << 4, -2, 1,
     -2, 4, -2,
     1, -2, 4;

PartialPivLU<Matrix3d> lu(A);
lu.matrixLU();            // L 与 U 压在一起（下三角为 L 的非对角部分）
lu.permutationP();        // 行置换 P
lu.determinant();         // 36
lu.inverse();             // 显式逆
lu.solve(b);

// 提取 L、U（注意 PartialPivLU 没有 matrixL()/matrixU() 成员！）
Matrix3d L = lu.matrixLU().triangularView<UnitLower>();
Matrix3d U = lu.matrixLU().triangularView<Upper>();
```

`FullPivLU` 提供更完整的接口：

```cpp
FullPivLU<MatrixXd> flu(A);
flu.rank();
flu.dimensionOfKernel();
flu.isInvertible();
flu.kernel();             // 核空间基（列向量）
flu.image();              // 像空间基
flu.inverse();
flu.determinant();
flu.matrixL();            // FullPivLU 有 matrixL()/matrixU()
flu.matrixU();
```

> 注意 `PartialPivLU` 只有 `matrixLU()`，`FullPivLU` 才有 `matrixL()`/`matrixU()`。这是最容易记混的一处。

### 12.2 Cholesky（LLT / LDLT）

仅用于对称（半）正定矩阵：

```cpp
Matrix3d S;
S << 4, 1, 1,
     1, 3, 0,
     1, 0, 2;

LLT<Matrix3d> llt(S);                     // A = L L^T
llt.matrixL().toDenseMatrix();
llt.matrixU().toDenseMatrix();
llt.solve(b);
llt.determinant();
llt.rcond();                              // 倒数条件数
llt.reconstructedMatrix();                // 重建 A
llt.info() == Eigen::Success;

LDLT<Matrix3d> ldlt(S);                   // A = L D L^T，更稳
ldlt.vectorD();                           // 对角 D
ldlt.matrixL();
ldlt.isPositive();                        // 是否正定
ldlt.isNegative();
ldlt.solve(b);
```

> 矩阵**不是** SPD 时，`LLT` 不会报错，只会给出错误结果。务必检查 `llt.info() == Eigen::Success`，或用 `ldlt().isPositive()` 验证。更保险的做法是先做 `SelfAdjointEigenSolver` 检查特征值符号。

### 12.3 QR 分解

```cpp
MatrixXd A2(4, 3);
A2 << 1, 2, 3,
      4, 5, 6,
      7, 8, 10,
      1, 1, 1;
VectorXd b2(4);
b2 << 1, 2, 3, 4;

// 1) HouseholderQR：最快，不选主元
HouseholderQR<MatrixXd> qr(A2);
VectorXd x = qr.solve(b2);
MatrixXd Q = qr.householderQ() * MatrixXd::Identity(4, 3);   // 取前 3 列（经济型 Q）
MatrixXd R = qr.matrixQR().topRows(3).triangularView<Upper>().toDenseMatrix();

// 2) ColPivHouseholderQR：列主元，能处理秩亏
ColPivHouseholderQR<MatrixXd> cpqr(A2);
cpqr.rank();
cpqr.dimensionOfKernel();
cpqr.colsPermutation().indices();     // 列置换
cpqr.matrixR();
cpqr.maxPivot();
cpqr.inverse();
VectorXd x2 = cpqr.solve(b2);

// 3) FullPivHouseholderQR：行列都选主元，最稳最慢
FullPivHouseholderQR<MatrixXd> fpqr(A2);

// 4) CompleteOrthogonalDecomposition：秩亏最小二乘 / 伪逆
CompleteOrthogonalDecomposition<MatrixXd> cod(A2);
cod.setThreshold(1e-10);
cod.pseudoInverse();
cod.solve(b2);
```

> `qr.householderQ()` 返回完整 Q（4×4）；乘以 `MatrixXd::Identity(4,3)` 得到经济型 Q（4×3）。

### 12.4 SVD

```cpp
MatrixXd R1(2, 3);
R1 << 1, 2, 3,
      2, 4, 7;

// JacobiSVD：精度最高，小矩阵首选
JacobiSVD<MatrixXd> svd(R1, ComputeThinU | ComputeThinV);
svd.singularValues();          // [9.10712, 0.24553]
svd.matrixU();                 // 2x2（Thin）
svd.matrixV();                 // 3x2（Thin）
svd.rank();                    // 2
svd.nonzeroSingularValues();
svd.threshold();
svd.setThreshold(1e-10);
svd.solve(b);                  // 最小二乘解

// 重建验证：A = U * S * V^T
MatrixXd rec = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();

// BDCSVD：大矩阵（>几百阶）更快
BDCSVD<MatrixXd> bdc(R1, ComputeThinU | ComputeThinV);
```

**关键限制**：`ComputeThinU` / `ComputeThinV` **只在列数为动态时可用**。若矩阵列数在编译期固定（如 `Matrix3d`），只能用 `ComputeFullU | ComputeFullV`，否则运行期断言失败：

```
Assertion `(!(m_computeThinU || m_computeThinV) || (MatrixType::ColsAtCompileTime==Dynamic))'
failed: "JacobiSVD: thin U and V are only available when your matrix has a dynamic number of columns."
```

条件数（用于判断病态程度）：

```cpp
double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
```

> 再次强调：`JacobiSVD` **没有** `conditionNumber()` 成员。

### 12.5 特征值分解

**对称矩阵**（`SelfAdjointEigenSolver`，只读三角，速度快、特征值实数且升序）：

```cpp
Matrix3d S;
S << 4, 1, 1,
     1, 3, 0,
     1, 0, 2;

SelfAdjointEigenSolver<Matrix3d> es(S);
es.eigenvalues();        // 升序：1.46791 2.6527 4.87939
es.eigenvectors();       // 列向量为对应特征向量
es.info();
es.operatorSqrt();       // 矩阵平方根
es.operatorInverseSqrt();

// 即用即取（返回特征值向量，不是矩阵）
Vector3d ev = S.selfadjointView<Lower>().eigenvalues();
```

> 若矩阵只是"数值上"对称，务必用 `selfadjointView<Lower>()` 或者自己保证对称，`SelfAdjointEigenSolver` 只读下三角（默认），上三角的内容会被忽略。

**广义特征值问题** `A v = λ B v`：

```cpp
GeneralizedSelfAdjointEigenSolver<Matrix3d> ges(A, B);
ges.eigenvalues();
ges.eigenvectors();
```

**一般实矩阵**（结果可能是复数）：

```cpp
EigenSolver<Matrix3d> ges2(A);
ges2.eigenvalues();              // complex<double> 向量
ges2.eigenvectors();             // 复矩阵
ges2.pseudoEigenvalueMatrix();   // 实伪特征值矩阵
ges2.pseudoEigenvectors();       // 实伪特征向量

// 复矩阵
ComplexEigenSolver<MatrixXcd> ces(A.cast<complex<double>>());

// 实 Schur 分解 / Hessenberg / 三对角化
RealSchur<Matrix3d> schur(A);
schur.matrixT();
schur.matrixU();

HessenbergDecomposition<Matrix3d> hess(A);
hess.matrixH();
hess.matrixQ();

Tridiagonalization<Matrix3d> tri(A);
```

### 12.6 分解类接口总览

| 类 | 关键成员 |
|---|---|
| `PartialPivLU` | `matrixLU()`, `permutationP()`, `solve()`, `inverse()`, `determinant()` |
| `FullPivLU` | 上面 + `matrixL()`, `matrixU()`, `rank()`, `kernel()`, `image()`, `isInvertible()`, `dimensionOfKernel()` |
| `LLT` | `matrixL()`, `matrixU()`, `solve()`, `determinant()`, `reconstructedMatrix()`, `rcond()`, `info()` |
| `LDLT` | `matrixL()`, `vectorD()`, `isPositive()`, `isNegative()`, `solve()`, `reconstructedMatrix()` |
| `HouseholderQR` | `matrixQR()`, `householderQ()`, `solve()` |
| `ColPivHouseholderQR` | 上面 + `rank()`, `matrixR()`, `colsPermutation()`, `maxPivot()`, `inverse()`, `dimensionOfKernel()` |
| `FullPivHouseholderQR` | 行列主元 + `rank()` |
| `CompleteOrthogonalDecomposition` | `pseudoInverse()`, `solve()`, `rank()`, `setThreshold()` |
| `JacobiSVD` / `BDCSVD` | `matrixU()`, `matrixV()`, `singularValues()`, `rank()`, `solve()`, `setThreshold()` |
| `SelfAdjointEigenSolver` | `eigenvalues()`, `eigenvectors()`, `operatorSqrt()`, `info()` |
| `EigenSolver` | `eigenvalues()`, `eigenvectors()`, `pseudoEigenvectors()` |
| `RealSchur` | `matrixT()`, `matrixU()` |

---

## 13. 稀疏矩阵

### 13.1 类型与声明

```cpp
#include <Eigen/Sparse>

SparseMatrix<double> A;                          // 默认列主序
SparseMatrix<double, RowMajor> B;                // 行主序
SparseVector<double> v;

SparseMatrix<double> M(1000, 1000);              // 声明尺寸
M.reserve(5);                                    // 预留每列非零元个数（重要！）
M.reserve(Eigen::VectorXi::Constant(1000, 5));   // 每列各自预留
```

### 13.2 构建：Triplet 列表（推荐）

逐个 `insert` 极慢；正确做法是先收集三元组，再一次性构建：

```cpp
int n = 6;
SparseMatrix<double> sp(n, n);
std::vector<Triplet<double>> trips;
trips.reserve(3 * n);

for (int i = 0; i < n; ++i) {
    trips.emplace_back(i, i, 4.0);
    if (i > 0)     trips.emplace_back(i, i - 1, -1.0);
    if (i + 1 < n) trips.emplace_back(i, i + 1, -1.0);
}
sp.setFromTriplets(trips.begin(), trips.end());   // 重复元素会自动求和
sp.makeCompressed();                              // 压缩存储（释放冗余空间）

std::cout << "nnz = " << sp.nonZeros() << "\n";   // 16
```

> `setFromTriplets` 对**相同位置的重复三元组自动累加**，这点非常适合有限元组装。
> 需要按规则合并时可用带 functor 的重载：`setFromTriplets(begin, end, dupFunctor)`。

### 13.3 元素访问与插入

```cpp
SparseMatrix<double> sp2(3, 3);
sp2.insert(0, 0) = 1.0;          // 仅用于首次插入，重复 insert 会产生重复项
sp2.coeffRef(2, 2) += 3.0;       // 读写；若不存在会插入（较慢）
sp2.makeCompressed();

double v = sp2.coeff(1, 1);      // 只读，不存在返回 0
// sp2.coeffRef(1, 1);           // 未压缩时不可用，需先 makeCompressed

sp2.setZero();                   // 清零但保留结构
sp2.prune(0.0);                  // 删除绝对值 <= 阈值的元素
sp2.setIdentity();
```

### 13.4 遍历与运算

```cpp
// 按外层（列主序时是列）遍历，最高效
for (int k = 0; k < sp.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(sp, k); it; ++it) {
        double val = it.value();
        int row = it.row(), col = it.col();
        it.valueRef() *= 2.0;      // 原地修改
    }

// 运算
sp * x;                 // 稀疏 × 稠密向量
sp * dense;             // 稀疏 × 稠密矩阵
sp.transpose();
sp.adjoint();
sp.block(0, 0, 3, 3);   // 稀疏块视图，只能整体赋值/参与运算，不能直接取 nonZeros()
sp.topLeftCorner(3, 3);
sp.diagonal();
sp.norm();
sp.sum();
```

> **打印注意**：`std::cout << sp` 输出的是内部存储结构（Nonzero entries / Outer pointers），不是矩阵。要按矩阵格式打印需转稠密：
> ```cpp
> std::cout << MatrixXd(sp) << "\n";    // 小矩阵可以，大矩阵慎用
> ```

### 13.5 稀疏求解器

```cpp
#include <Eigen/SparseCholesky>              // SimplicialLLT / SimplicialLDLT
#include <Eigen/SparseLU>                    // SparseLU
#include <Eigen/SparseQR>                    // SparseQR
#include <Eigen/IterativeLinearSolvers>      // CG / BiCGSTAB

// 1) 对称正定：SimplicialLLT（直接法，最常用）
SimplicialLLT<SparseMatrix<double>> llt;
llt.compute(sp);
if (llt.info() != Eigen::Success) { /* 分解失败，矩阵非正定 */ }
VectorXd x = llt.solve(b);

SimplicialLDLT<SparseMatrix<double>> ldlt;   // 半正定/不定也能用
ldlt.compute(sp);

// 2) 一般方阵：SparseLU
SparseLU<SparseMatrix<double>> slu;
slu.analyzePattern(sp);        // 符号分析（结构不变时可复用）
slu.factorize(sp);             // 数值分解
VectorXd x2 = slu.solve(b);
slu.determinant();
slu.logAbsDeterminant();

// 3) 最小二乘：SparseQR
SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> sqr;
sqr.compute(sp);
VectorXd x3 = sqr.solve(b);
sqr.rank();

// 4) 迭代法：共轭梯度（SPD）
ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
cg.setTolerance(1e-12);
cg.setMaxIterations(100);
cg.compute(sp);
VectorXd x4 = cg.solve(b);
std::cout << cg.iterations() << " " << cg.error() << "\n";
```

迭代求解器对比：

| 求解器 | 适用矩阵 | 备注 |
|---|---|---|
| `ConjugateGradient` | SPD | 需要 `Lower\|Upper`（默认只用下三角） |
| `BiCGSTAB` | 一般方阵 | 不稳定，但对流固耦合类问题有效 |
| `LeastSquaresConjugateGradient` | 长方形/最小二乘 | — |
| `GMRES` | 一般方阵 | `#include <unsupported/Eigen/IterativeSolvers>` |

加速：加预条件器，如 `IncompleteCholesky<double>`：

```cpp
ConjugateGradient<SparseMatrix<double>, Lower | Upper, IncompleteCholesky<double>> cg2;
```

### 13.6 稀疏 ↔ 稠密转换

```cpp
MatrixXd dense = MatrixXd::Zero(3, 3);
dense(0, 0) = 1; dense(1, 1) = 2;

SparseMatrix<double> fromDense = dense.sparseView();          // 稠密 → 稀疏
SparseMatrix<double> pruned = dense.sparseView(1.0, 1e-6);    // (reference, epsilon)：|x| <= eps*|ref| 视为 0

MatrixXd back = MatrixXd(sp);                                  // 稀疏 → 稠密
MatrixXd back2(sp);                                            // 同上
```

### 13.7 稀疏性能要点

1. **始终用 Triplet 构建**，不要循环 `insert` / `coeffRef`（O(nnz) 每次）。
2. **先 `reserve`** 再填充，避免反复重分配。
3. `makeCompressed()` 之后再读取/运算更快。
4. 稀疏矩阵的 `block()` 返回 `SparseBlock` 视图，`nonZeros()` 等成员不可访问，需要先赋给 `SparseMatrix` 再用。
5. 求解前用 `AMDOrdering` / `COLAMDOrdering` 重排可显著减少填充（`SparseLU`/`SparseQR` 默认已用）。

---
## 14. 几何模块

```cpp
#include <Eigen/Geometry>
```

### 14.1 旋转的四种表示与互转

| 类型 | 说明 | 存储 |
|---|---|---|
| `Matrix3d` | 旋转矩阵 | 9 个数 |
| `AngleAxisd` | 轴角（角度 + 单位轴） | 4 个数 |
| `Quaterniond` | 四元数 | 4 个数 |
| `Rotation2Dd` | 2D 旋转 | 1 个角度 |

```cpp
// 轴角 -> 矩阵
Matrix3d R = AngleAxisd(M_PI / 4, Vector3d::UnitZ()).toRotationMatrix();

// 矩阵 -> 四元数 -> 轴角
Quaterniond q(R);
AngleAxisd aa(q);

q.w(); q.x(); q.y(); q.z();       // 标量 + 虚部
q.coeffs();                       // 注意顺序是 (x, y, z, w)！
Quaterniond q2(0.92388, 0, 0, 0.382683);   // 构造顺序是 (w, x, y, z)

aa.angle();                       // 0.785398
aa.axis();                        // 0 0 1

// 欧拉角：参数是轴顺序（2=Z, 1=Y, 0=X），返回 (yaw, pitch, roll)
Vector3d ypr = R.eulerAngles(2, 1, 0);     // 0.785398, 0, 0
```

> 从旋转矩阵构造四元数时，Eigen **假设输入矩阵严格正交**。若矩阵有数值误差，四元数不会自动归一化（实测 `q.norm() = 1.00187`）。务必：
> ```cpp
> Quaterniond q(M);
> q.normalize();
> ```
> 或用 `R.isUnitary()` 检查；严重不满足时先做 SVD 正交化。

### 14.2 四元数运算

```cpp
Vector3d v(1, 0, 0);

Quaterniond q1(AngleAxisd(0.3, Vector3d::UnitX()));
Quaterniond q2(AngleAxisd(0.5, Vector3d::UnitY()));

q1 * q2;                 // 复合旋转（注意顺序：先 q2 后 q1）
q1.normalize();          // 归一化
q1.normalized();         // 返回归一化副本
q1.conjugate();          // 共轭（= 逆，对单位四元数）
q1.inverse();
q1.slerp(0.5, q2);       // 球面线性插值，t=0.5 时取中间
q1.toRotationMatrix();
q1 * v;                  // 直接旋转向量（要求 q1 为单位四元数）

// 从两个向量构造旋转
Quaterniond qv = Quaterniond::FromTwoVectors(Vector3d(1,0,0), Vector3d(0,1,0));
```

> 四元数乘法**不可交换**；`q1 * q2` 表示"先应用 q2，再应用 q1"（与矩阵乘法一致）。

### 14.3 Transform：旋转 + 平移

| 类型 | 含义 | 自由度 |
|---|---|---|
| `Transform<double, 3, Affine>` / `Affine3d` | 仿射（旋转+平移+缩放+剪切） | 12 |
| `Isometry3d` | 等距（旋转+平移） | 6 |
| `Projective3d` | 投影变换 | 15 |
| `Transform<double, 2, Affine>` | 2D 仿射 | 6 |

```cpp
Isometry3d T = Isometry3d::Identity();
T.rotate(AngleAxisd(M_PI / 2, Vector3d::UnitZ()));   // 先设置旋转
T.pretranslate(Vector3d(1, 2, 3));                   // 再设置平移

T.matrix();                     // 4x4 齐次矩阵
T.linear();                     // 3x3 旋转部分
T.translation();                // 3x1 平移部分
T * Vector3d(1, 0, 0);          // 变换点：(1, 3, 3)
T.inverse();
T.inverse() * T;                // 单位变换

// 常见取法
T.rotation();                   // 线性部分，返回 3x3 矩阵（可用 Quaternion 赋值）
Matrix4d M = T.matrix();
Isometry3d T2(M);               // 从 4x4 矩阵构造（会自动检查正交性）
```

> `pretranslate` / `prerotate` 是**左乘**（在现有变换之前施加，即在世界坐标系下），`translate` / `rotate` 是**右乘**（在局部坐标系下）。用错会导致完全不同的结果：

```cpp
Transform<double, 3, Affine> T1 = Transform<double, 3, Affine>::Identity();
T1.translate(Vector3d(1, 0, 0));                        // 右乘：先平移
T1.rotate(AngleAxisd(M_PI / 2, Vector3d::UnitZ()));     // 再绕“新的”局部 Z 轴转
T1 * Vector3d(0, 0, 0);                                 // (1, 0, 0)
```

### 14.4 2D 几何

```cpp
Rotation2Dd r2(M_PI / 6);
r2.toRotationMatrix();     // 2x2
r2 * Vector2d(1, 0);       // (0.866025, 0.5)
r2.smallestAngle();

Transform<double, 2, Affine> T2 = Transform<double, 2, Affine>::Identity();
T2.translate(Vector2d(1, 2));
T2.rotate(r2);
```

### 14.5 常见任务片段

```cpp
// 求两点间变换
Vector3d p1(0,0,0), p2(1,2,3);
Isometry3d Tt = Isometry3d::Identity();
Tt.translation() = p2 - p1;

// 绕任意轴旋转
Vector3d axis(1, 1, 0); axis.normalize();
Matrix3d Rr = AngleAxisd(0.7, axis).toRotationMatrix();

// 从旋转矩阵提取欧拉角（ZYX 顺序）
Vector3d euler = Rr.eulerAngles(2, 1, 0);

// 位姿插值（平移线性 + 旋转球面插值）
// 假设已有旋转矩阵 R1、R2 与平移 t1、t2
Quaterniond qa(R1), qb(R2);
Quaterniond qi = qa.slerp(0.3, qb);
Vector3d ti = t1 + 0.3 * (t2 - t1);
```

---

## 15. Map 与 Ref：外部内存与函数参数

### 15.1 Map：零拷贝包装外部内存

```cpp
double raw[6] = {1, 2, 3, 4, 5, 6};

// 固定尺寸，默认列主序
Map<Matrix<double, 2, 3>> m1(raw);
// 1 3 5
// 2 4 6

// 行主序解释同一块内存
Map<Matrix<double, 2, 3, RowMajor>> m2(raw);
// 1 2 3
// 4 5 6

// 动态尺寸
Map<MatrixXd> m3(raw, 3, 2);
m3(0, 0) = 100;                  // 直接修改 raw[0]；m1、m2、m3 共享同一块内存

// 只读
const double* craw = raw;
Map<const VectorXd> v1(craw, 6);

// 与 std::vector 互操作
std::vector<double> buf = {1, 2, 3, 4};
Map<VectorXd> mv(buf.data(), buf.size());
mv *= 2;                        // 直接改 buf
```

> `Map` 不做内存拷贝，也不管理内存。**外部缓冲区必须比 Map 活得久**，且尺寸/存储顺序必须严格匹配。

### 15.2 对齐问题

若外部缓冲区未按 Eigen 要求对齐（16/32 字节），用 `Unaligned` 选项：

```cpp
Map<Matrix<double, 2, 2>, Unaligned> mu(raw);
```

一般情况不需要；只在明确知道指针未对齐（如来自第三方库）时才用，会有性能损失。

### 15.3 Stride：处理带 padding 的缓冲区

图像/BLAS 缓冲区常见"每行有填充"，例如每行 5 个 `double` 但只用前 3 个：

```cpp
std::vector<double> buf(15);
for (int i = 0; i < 15; ++i) buf[i] = i;

// 行主序，行跨度 5（OuterStride）
Map<Matrix<double, 3, 3, RowMajor>, 0, OuterStride<5>> img(buf.data());
// 0  1  2
// 5  6  7
// 10 11 12
```

`Stride<Outer, Inner>`：`Outer` 是相邻行（行主序）或相邻列（列主序）之间的跨距，`Inner` 是行/列内部相邻元素的跨距。

### 15.4 Ref：函数参数的正确写法

写接受 Eigen 对象的函数时，**不要用 `const MatrixXd&`**（会强制调用方构造临时对象），用 `Ref`：

```cpp
// 可接受 MatrixXd、block、Map，甚至表达式（表达式会先求值到临时）
double frob(const Ref<const MatrixXd>& m) { return m.norm(); }

void scale_inplace(Ref<MatrixXd> m, double s) { m *= s; }

MatrixXd A = MatrixXd::Random(3, 3);
frob(A);                          // OK
frob(A.topLeftCorner(2, 2));      // OK：块
frob(2 * A);                      // OK：表达式，内部构造临时
scale_inplace(A, 0.5);            // OK：原地修改
```

| 写法 | 能接受块/Map | 会拷贝 |
|---|---|---|
| `const MatrixXd&` | 否（需转换，临时） | 是（当传入块时） |
| `const Ref<const MatrixXd>&` | 是 | 否 |
| `Ref<MatrixXd>` | 是 | 否（可写） |
| 模板 `template<class Derived> f(const MatrixBase<Derived>&)` | 是 | 否（但代码膨胀） |

---

## 16. 输入输出与调试

### 16.1 打印

```cpp
MatrixXd M = MatrixXd::Random(3, 3);
std::cout << M << "\n";                          // 默认格式
std::cout << M.transpose() << "\n";
std::cout << std::setprecision(10) << M << "\n"; // 配合 <iomanip>
```

### 16.2 自定义格式

```cpp
// IOFormat(精度, 对齐标志, 元素分隔, 行分隔, 行前缀, 行后缀, 矩阵前缀, 矩阵后缀)
IOFormat fmt(4, 0, ", ", "\n", "[", "]");
std::cout << M.format(fmt) << "\n";
// [-41.84, -59.79, 8.913]
// [-41.02, -80.93, -24.42]
// ...

// CSV
std::cout << M.format(IOFormat(StreamPrecision, DontAlignCols, ",", "\n"));

// MATLAB / Octave 可直接粘贴
std::cout << M.format(IOFormat(FullPrecision, DontAlignCols, ",", ";\n", "", "", "[", "]"));

// 不打印矩阵括号（便于写入文件）
std::cout << M.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "", ""));
```

常用标志：`DontAlignCols`、`StreamPrecision`、`FullPrecision`。

### 16.3 读写文件

```cpp
#include <fstream>

// 写文本
{
    std::ofstream os("mat.txt");
    os << M.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n"));
}

// 读文本
MatrixXd N(3, 3);
{
    std::ifstream is("mat.txt");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) is >> N(i, j);
}
std::cout << N.isApprox(M) << "\n";   // 1
```

二进制读写（要求同一架构、同一存储顺序）：

```cpp
{
    std::ofstream os("mat.bin", std::ios::binary);
    os.write(reinterpret_cast<const char*>(M.data()), M.size() * sizeof(double));
}
```

### 16.4 断言与调试

Eigen 内部用 `eigen_assert`，在 debug 下检查尺寸、越界、别名等。

```cpp
#define EIGEN_NO_DEBUG          // 关闭所有断言（发布版自动生效于 NDEBUG）
#define EIGEN_NO_AUTOMATIC_RESIZING   // 禁止自动 resize，尺寸不匹配直接断言
```

发布构建（`-DNDEBUG`）下断言自动关闭。**开发阶段不要定义 `EIGEN_NO_DEBUG`**，否则尺寸错误会变成静默的内存越界。

---

## 17. 性能实践

### 17.1 编译选项

```bash
g++ -O3 -DNDEBUG -march=native -std=c++17 -I/usr/include/eigen3 main.cpp
```

- `-O3`：更激进的向量化与循环展开，矩阵乘等 kernel 明显更快（`-O2` 也能向量化，但部分 kernel 不展开）
- `-DNDEBUG`：关闭断言
- `-march=native`：启用 SSE/AVX/FMA，矩阵乘可快 2~4 倍
- `-fopenmp`：启用 Eigen 内置的多线程（大矩阵乘法、部分归约）

> `-march=native` 编译出的二进制不可跨机器分发；AVX 要求 32 字节对齐，若与第三方未对齐缓冲区交互会出现段错误，可用 `-DEIGEN_MAX_ALIGN_BYTES=16` 缓解。

### 17.2 内存对齐规则

固定尺寸、且大小是 16 字节整数倍的 Eigen 类型（如 `Vector4d`、`Matrix4d`、`Vector2d`）要求对齐。

```cpp
// 1) 作为类成员：C++17 前必须加宏
struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector4d p;
    Eigen::Matrix4d T;
};

// 2) 放进 STL 容器：始终用 aligned_allocator
std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pts;

// 3) 结构体数组：C++17 前需要 aligned_allocator 或手动对齐分配
```

C++17 起 `operator new` 保证按 `__STDCPP_DEFAULT_NEW_ALIGNMENT__`（通常 16 字节）对齐，SSE 场景基本安全；但 AVX（32 字节）/AVX512（64 字节）下 `std::vector` 仍可能不对齐，**建议始终显式使用 `aligned_allocator`**。

### 17.3 编码建议

| 建议 | 说明 |
|---|---|
| 用固定尺寸处理小矩阵 | `Matrix4d` 比 `MatrixXd(4,4)` 快数倍 |
| 避免 `auto` 接表达式 | 用具体类型或 `.eval()` |
| 大矩阵乘用 `noalias()` | `C.noalias() = A * B;` 省一次临时 |
| 复用分解对象 | 同一矩阵多次求解时只分解一次 |
| 循环内不 `resize` | 提前分配好 |
| 用 `.array()` 做逐元素运算 | 保证向量化 |
| 按存储顺序遍历 | 列主序按列遍历 |
| 避免 `A.inverse() * b` | 改用 `solve()` |
| 稀疏用 Triplet 构建 | 比逐个 insert 快百倍 |
| 热循环里少用 `block()` 链式表达式 | 生成复杂表达式树，可能不向量化 |

### 17.4 并行

```cpp
// 大矩阵乘法、部分归约会自动多线程（需 -fopenmp）
// 控制线程数
Eigen::setNbThreads(4);   // 需要 -fopenmp 编译，否则返回 1
Eigen::nbThreads();

// 关闭并行
// -DEIGEN_DONT_PARALLELIZE
```

---
## 18. 常见坑速查

按"踩中概率"排序，前几条几乎人人都会遇到。

| # | 现象 | 原因 | 正确做法 |
|---|---|---|---|
| 1 | `VectorXd v(1, 2)` 编译报 `THIS_METHOD_IS_ONLY_FOR_VECTORS_OF_A_SPECIFIC_SIZE` | 动态向量没有 `(a,b)` 元素构造，只有 `(size)` | `VectorXd v(2); v << 1, 2;` |
| 2 | `auto x = A + B;` 后结果乱码/崩溃 | `auto` 捕获表达式，引用了已销毁的临时对象 | `MatrixXd x = A + B;` 或 `auto x = (A+B).eval();` |
| 3 | `A = A.transpose()` 运行期断言 `aliasing detected` | 转置赋值存在别名 | `A.transposeInPlace()`（方阵）或 `A = A.transpose().eval()` |
| 4 | `M.rowwise() - M.rowwise().mean()` 编译失败 | `rowwise()` 要求右侧是**行向量** | `M - M.rowwise().mean().replicate(1, M.cols())` |
| 5 | `M.rowwise() - 列向量` 编译报 `YOU_PASSED_A_COLUMN_VECTOR...` | 广播规则反直觉 | `rowwise` 配 1×N，`colwise` 配 N×1 |
| 6 | `MatrixXd m; m(0,0)=1;` 崩溃 | 动态矩阵默认 0×0 | 先 `m.resize(r,c)` 或 `MatrixXd m(r,c)` |
| 7 | `cout << A.triangularView<Lower>()` 编译失败 | 三角/自伴视图不是 `DenseBase` | `.toDenseMatrix()` 或先赋给矩阵 |
| 8 | `PartialPivLU::matrixL()` 不存在 | 只有 `matrixLU()` | `lu.matrixLU().triangularView<UnitLower>()` |
| 9 | `JacobiSVD` 用 `ComputeThinU` 断言失败 | Thin 只支持列数动态 | 固定尺寸用 `ComputeFullU|ComputeFullV` |
| 10 | `svd.conditionNumber()` 不存在 | JacobiSVD 无此成员 | `s(0)/s(s.size()-1)` |
| 11 | `cout << sparseMat` 输出一堆 `Nonzero entries` | 稀疏矩阵的 `operator<<` 打印内部结构 | `cout << MatrixXd(sp)` |
| 12 | 从旋转矩阵构造的四元数 `norm() != 1` | 输入矩阵非严格正交 | 构造后 `q.normalize()` |
| 13 | `Matrix3d m = matrixXd;` 编译通过但运行期断言 | 动态→固定只在运行期检查尺寸 | 确认尺寸一致，或全程使用 `MatrixXd` |
| 14 | `Eigen::Vector4d` 放进 `std::vector` 段错误 | 未对齐分配 | `std::vector<Vector4d, Eigen::aligned_allocator<Vector4d>>` |
| 15 | 循环里 `insert` 构建稀疏矩阵极慢 | 每次插入 O(nnz) | 用 `Triplet` + `setFromTriplets` |
| 16 | 矩阵乘结果全 0 / 崩溃 | 维度不匹配，或用了 `cwiseProduct` 当矩阵乘 | 检查 `A.cols() == B.rows()`；矩阵乘用 `*` |
| 17 | `m(i,j)` 在 release 下越界不报错 | `NDEBUG` 关闭断言 | 关键路径手动检查尺寸 |
| 18 | `LLT` 对非正定矩阵给出错误结果且不报错 | LLT 不做检查 | 查 `llt.info()` 或改用 `ldlt().isPositive()` |
| 19 | `x.transpose()*A*x` 无法赋给 `double` | 结果是 1×1 矩阵 | 用 `.value()` 或 `(expr)(0,0)` |
| 20 | 固定尺寸大矩阵（如 `Matrix<double,200,200>`）栈溢出 | 栈上分配 | 改用 `MatrixXd` |

补充说明几条：

```cpp
// 13 的细节
MatrixXd big = MatrixXd::Random(3, 3);
Matrix3d fixed = big;          // 固定尺寸赋值：运行期断言尺寸，编译期不报错
// Matrix3d bad = MatrixXd::Random(4,4);  // 运行期断言失败

// 19 的细节
VectorXd x = VectorXd::Ones(3);
double q = (x.transpose() * Matrix3d::Identity() * x)(0, 0);
// 或
double q2 = x.dot(Matrix3d::Identity() * x);
```

---

## 19. API 速查表

### 19.1 尺寸与属性

| API | 说明 |
|---|---|
| `A.rows()` / `A.cols()` | 行数 / 列数 |
| `A.size()` | 元素总数 |
| `A.innerSize()` / `A.outerSize()` | 内/外维度（列主序：行内、列外） |
| `A.data()` | 裸指针 |
| `A.size()==0` | 空矩阵判断 |
| `A.nonZeros()` | 稀疏矩阵非零元个数 |

### 19.2 生成器

| API | 说明 |
|---|---|
| `MatrixXd::Zero(r,c)` / `Ones` / `Constant(r,c,v)` / `Identity(r,c)` / `Random(r,c)` | 静态生成 |
| `VectorXd::LinSpaced(n, lo, hi)` | 等距序列 |
| `MatrixXd::NullaryExpr(r,c,f)` | 函数生成 |
| `A.setZero()` / `setOnes()` / `setIdentity()` / `setConstant(v)` / `setRandom()` | 原地设置 |

### 19.3 运算

| API | 说明 |
|---|---|
| `A * B` / `A * v` / `s * A` | 矩阵乘 |
| `A.cwiseProduct(B)` / `cwiseQuotient` | 逐元素 |
| `A.array()` / `A.matrix()` | 语义切换 |
| `A.transpose()` / `adjoint()` / `conjugate()` | 转置/共轭转置/共轭 |
| `A.dot(v)` / `cross(v)` / `outer(v)` | 点积/叉积/外积 |
| `A.norm()` / `squaredNorm()` / `lpNorm<p>()` | 范数 |
| `A.sum()` / `mean()` / `prod()` / `minCoeff()` / `maxCoeff()` / `trace()` | 归约 |
| `A.colwise()` / `rowwise()` | 按列/按行 |
| `A.unaryExpr(f)` / `binaryExpr(B,f)` | 自定义逐元素 |
| `A.cast<T>()` | 类型转换 |
| `A.eval()` / `noalias()` | 强制求值 / 免临时 |
| `A.isApprox(B, tol)` | 近似相等 |

### 19.4 分解与求解

| API | 说明 |
|---|---|
| `A.partialPivLu()` / `fullPivLu()` | LU |
| `A.llt()` / `ldlt()` | Cholesky |
| `A.householderQr()` / `colPivHouseholderQr()` / `fullPivHouseholderQr()` | QR |
| `A.completeOrthogonalDecomposition()` | COD（伪逆） |
| `A.jacobiSvd()` / `bdcSvd()` | SVD |
| `A.eigenvalues()` | 自伴矩阵特征值（便捷接口，无 `eigenvectors()` 版本） |
| `A.selfadjointView<Lower>()` | 自伴视图 |
| `A.triangularView<Upper>()` | 三角视图 |
| `A.inverse()` / `determinant()` / `trace()` | 逆/行列式/迹 |

### 19.5 常用 include 对照

| 想用 | 需要 include |
|---|---|
| `MatrixXd`, `Vector3d` | `<Eigen/Core>` 或 `<Eigen/Dense>` |
| `llt()`, `ldlt()` | `<Eigen/Cholesky>` |
| `partialPivLu()`, `fullPivLu()` | `<Eigen/LU>` |
| `householderQr()` 等 | `<Eigen/QR>` |
| `jacobiSvd()` | `<Eigen/SVD>` |
| `SelfAdjointEigenSolver` | `<Eigen/Eigenvalues>` |
| `Quaterniond`, `Isometry3d` | `<Eigen/Geometry>` |
| `SparseMatrix` | `<Eigen/Sparse>` |
| `ConjugateGradient` | `<Eigen/IterativeLinearSolvers>` |
| `SimplicialLLT` / `SimplicialLDLT` | `<Eigen/SparseCholesky>` |
| `SparseLU` | `<Eigen/SparseLU>` |
| `SparseQR` | `<Eigen/SparseQR>` |

---

## 20. 常见编译错误解读

Eigen 的模板报错动辄几百行。**只看 `static assertion failed:` 后面那句全大写英文**即可定位。

| 错误信息 | 原因 | 修复 |
|---|---|---|
| `YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES` | 固定尺寸不匹配，如 `Matrix3d = Matrix4d` | 统一尺寸或改动态 |
| `YOU_MIXED_DIFFERENT_NUMERIC_TYPES` | `double` 与 `float` 混算 | 显式 `.cast<float>()` |
| `THIS_METHOD_IS_ONLY_FOR_VECTORS_OF_A_SPECIFIC_SIZE` | 对动态向量用了 `(a,b,c)` 构造 | 改 `v << a, b, c;` |
| `YOU_PASSED_A_COLUMN_VECTOR_BUT_A_ROW_VECTOR_WAS_EXPECTED` | `rowwise()` 配了列向量 | 换行向量或转置 |
| `YOU_PASSED_A_ROW_VECTOR_BUT_A_COLUMN_VECTOR_WAS_EXPECTED` | `colwise()` 配了行向量 | 换列向量或转置 |
| `THE_BRACKET_OPERATOR_IS_ONLY_FOR_VECTORS__USE_THE_PARENTHESIS_OPERATOR_INSTEAD` | 对矩阵用了 `A[i]` | 改 `A(i,j)` |
| `INVALID_MATRIX_TEMPLATE_PARAMETERS` | `Matrix` 模板参数非法（Max 小于实际等） | 检查 `RowMajor`/Max 参数 |
| `INVALID_MATRIX_PRODUCT` | 矩阵乘维度编译期不匹配 | 检查 `A.cols()==B.rows()` |
| `YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR` | 对动态尺寸调用了 `x()`/`y()` 等 | 用 `v(0)` |
| `THIS_FUNCTION_IS_NOT_FOR_INTEGER_NUMERIC_TYPES` | 对整型矩阵求逆/SVD | 转 `double` |
| `YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX` | 对矩阵调用 `dot`/`cross` | 先取列 `A.col(i).dot(...)` |
| `undefined reference to Eigen::internal::...` | 极少见，通常是混用了不同版本头文件 | 统一 Eigen 版本，清理 build 目录 |
| `no match for operator<<` + `TriangularView`/`SelfAdjointView` | 视图不可直接打印 | `.toDenseMatrix()` |
| 编译时间极长 / 内存爆 | 表达式树过深 | 中间结果用具体类型变量"断链" |

**读报错的技巧**：

1. 从**最后一条** `required from here` 往上找，第一个 `error:` 通常才是真因；
2. 搜索 `YOU_`、`THIS_`、`INVALID_` 关键字；
3. 表达式链过长时，逐段加 `.eval()` 或赋给临时变量，报错会立刻变得可读。

```cpp
// 报错不可读时，拆开写
MatrixXd t1 = A + B;         // 若这里有错，问题在 A+B
MatrixXd t2 = t1 * C;
MatrixXd r  = t2.transpose();
```

---

## 附录 A：一个完整的可编译示例

演示 PCA（主成分分析）的核心步骤，串联了本文大部分概念。

```cpp
#include <Eigen/Dense>
#include <iostream>

int main() {
    // 100 个 3 维样本（列主序：3 行 100 列）
    const int N = 100, D = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(D, N);

    // 1) 中心化。这里每列是一个样本（D×N 布局），
    //    所以用 colwise() 配 D 维列向量，逐列减去均值向量；
    //    若数据是 N×D（每行一个样本），则写成 X.rowwise() - X.colwise().mean()
    Eigen::MatrixXd Xc = X.colwise() - X.rowwise().mean();

    // 2) 协方差矩阵（无偏）
    Eigen::MatrixXd cov = (Xc * Xc.transpose()) / double(N - 1);

    // 3) 特征分解（协方差对称，用 SelfAdjointEigenSolver）
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    // 特征值升序，取最大的前 k 个需要反序
    Eigen::VectorXd evals = es.eigenvalues().reverse();
    Eigen::MatrixXd evecs = es.eigenvectors().rowwise().reverse();

    std::cout << "特征值（降序）:\n" << evals.transpose() << "\n";

    // 4) 投影到前 2 个主成分
    int k = 2;
    Eigen::MatrixXd W = evecs.leftCols(k);      // D x k
    Eigen::MatrixXd Y = W.transpose() * Xc;     // k x N
    std::cout << "投影后形状: " << Y.rows() << "x" << Y.cols() << "\n";

    // 5) 重建误差
    Eigen::MatrixXd Xrec = W * Y;
    std::cout << "重建相对误差: " << (Xrec - Xc).norm() / Xc.norm() << "\n";
    return 0;
}
```

编译：

```bash
g++ -std=c++17 -O3 -DNDEBUG -I/usr/include/eigen3 pca.cpp -o pca && ./pca
```

## 附录 B：版本差异提示

| 特性 | 最低版本 |
|---|---|
| `reshaped()` | 3.4 |
| `A.eigenvalues()` 便捷接口 | 3.3 |
| 范围 for 循环遍历矩阵 | 3.4 |
| `CompleteOrthogonalDecomposition` | 3.3 |
| `isConstant(value)` 带参数版本 | 3.4 |
| `LinSpaced` 的 `Sequential_t` 版本 | 3.4 起废弃 |
| `Eigen::seq` / `Eigen::all` 切片语法 | **尚未提供**（3.4 及以前） |

## 参考

- 官方文档：https://eigen.tuxfamily.org/dox/
- 快速参考：https://eigen.tuxfamily.org/dox/QuickRefPage.html
- 源码仓库：https://gitlab.com/libeigen/eigen
- Eigen 与 STL 容器：https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
- 对齐问题（固定尺寸可向量化类型）：https://eigen.tuxfamily.org/dox/group__TopicFixedSizeVectorizable.html

---
layout: post
title: Matrix Decomposition
subtitle: Decompositions for Solving Least Squares
cover-img: /assets/img/2021-01-29-matrix-decomposition/kelly-sikkema.jpg
thumbnail-img: /assets/img/2021-01-29-matrix-decomposition/what-is-decomposition.png
readtime: true
show-avatar: false
tags: [Python, MATLAB, LinearAlgebra]
comments: true
---

The most important application for decomposition is in data fitting. The following discussion is mostly presented in terms of different methods of decomposition for linear function.

# Introduction

The method of least squares is a standard approach in regression analysis to approximate the solution of overdetermined systems (sets of equations in which there are more equations than unknowns) by minimizing the sum of the squares of the residuals made in the results of every single equation. An matrix decomposition is a way of reducing a matrix into its constituent parts. It's an approach that can specify more complex matrix operation that can be performed on the decomposed matrx rather than on the origin matrix itself. There are various matrix decomposition methods, such as LU decomposition, QR decomposition, SVD decomposition, and Cholesky decomposition, etc.

# LU Decomposition

> Least Square: Let $X \in \mathbb{R}^{m \times n}$ with $m>n$, $y \in \mathbb{R}^{m \times 1}$, and $\beta \in \mathbb{R}^{n \times 1}$. We aim to solve $y=X \beta$ where $\hat{\beta}$ is the least square estimator. The least squares solution for $\hat{\beta}=(X^{T} X)^{-1}X^{T}y$ can obtained using different decomposition methods on $X^{T} X$.

When using LU, we have $\hat{\beta}=(X^{T} X)^{-1}X^{T}y=(LU)^{-1}X^{T}y$, decomposing the square matrix $X^{T} X$ into $L$ and $U$ components. The factors $L$ and $U$ are triangular matrices. A variation of this decomposition that is numerically more stable to solve in practice is called the PLU decomposition, or the LU decomposition with partial pivoting, where $P$ is a so-called permutation matrix, $L$ is lower triangular, and $U$ is upper triangular. Lower and upper triangular matrices are computationally easier than your typical invertible matrix. The matrix P is easy to deal with as well since it is mostly full of zeros.  This [video](https://www.youtube.com/watch?v=UlWcofkUDDU&ab_channel=Mathispower4u) explains how to find the LU decomposition of a square matrix using a shortcut involving the opposite of multipliers used when performing row operations. There is also another [posting](https://math.unm.edu/~loring/links/linear_s08/LU.pdf) describe LU and PLU factorisation with some examples.

## Pseudocode

**Step 1.** Start with three candidate matrices:

* $U = M$
* $L = 0_{n, n}$
* $P^{T} = I_{n}$

where $L$ is a $n \times n$ zeros matrix and $P_{T}$ is a $n \times n$ identity matrix.

**Step 2.** For $i=1, 2, \ldots, n-1$, find the row $j$ with the largest entry in absolute value on or below the diagonal of the $i$-row and swap rows $i$ and $j$ in all three matrices, $P_{T}$, $L$, and $U$. If this maximum entry is zero, then terminate this loop and indicate that the matrix is singular (invertible).

**Step 3.** Inside the first loop, create a second for loop, for $j=i+1, \ldots, n$, calculate the scalar value $s= \frac{-u_{j, i}}{u_{i, i}}$. Next, add $s$ times row $i$ onto row $j$ in the matrix $U$ and set the entry $L_{j, y}=-s$.

**Step 4.** Having iterated from $ i=1, 2, \ldots, n-1$, finish by adding the identity matrix onto $L=L+I_{n}$. These are the $P_{T}$, $L$, and $U$ matrices of the PLU decomposition of matrix $M$.

```python
class Decomposition:
    """
    References
    ----------
    [1] https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/04LinearAlgebra/lup/
    [2] https://math.unm.edu/~loring/links/linear_s08/LU.pdf
    [3] https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
    """
    def __init__(self):
        pass
    
    def plu(self, A):
        # Step 1. Inittiate three cadidate matrices
        n = A.shape[0]
        P = np.identity(n)
        L = np.identity(n)
        U = A.copy()
        
        PF = np.identity(n)
        LF = np.zeros(shape=(n, n))
        
        # Step 2. Loop over rows find the row with the largest entry in absolute
        # value on or below the diagonal of the i-th row
        for i in range(n-1):
            index = np.argmax(abs(U[i:, i]))
            index = index + i
            if index != i:
                P = np.identity(n)
                P[[index, i], i:n] = P[[i, index], i:n]
                U[[index, i], i:n] = U[[i, index], i:n] 
                PF = np.dot(P, PF)
                LF = np.dot(P, LF)
            L = np.identity(n)
            # Step 3. Calculate the scalar value
            for j in range(i+1, n):
                L[j, i]  = -(U[j, i] / U[i, i])
                LF[j, i] =  (U[j, i] / U[i, i])
            U = np.dot(L, U)
        # Step 4. Add identity matrix onto L
        np.fill_diagonal(LF, 1)
        return PF, LF, U
```

# QR decomposition

In linear algebra, a QR decomposition, also known as a QR factorization or QU factorization is a decomposition of a matrix A, either square or rectangular, into a product $A = QR$ of an orthogonal matrix $Q$ and an upper triangular matrix $R$. There are in fact a couple of methods to compute a QR decomposition. These include the the `Gram–Schmidt process`, `Householder transformations`, and `Givens rotations`.

If $A \in \mathbb{R^{m \times n}}$ has linearly independent columns, then it can factored as: 

<div>
    <span>
        \[
            A= 
            \left[ {\begin{array}{c}
            q_{1} & q_{2} & \ldots & q_{n} \\
            \end{array} } \right]
            \left[ {\begin{array}{ccc}
            R_{11} & R_{12} & \ldots & R_{1n}\\
            0 & R_{22} & \ldots & R_{2n} \\
            \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & \vdots & R_{nn} \\
            \end{array} } \right]
        \]
    </span>
</div>

where $q_{1}, q_{2}, \ldots, q_{n}$ are orthogonal $m$-vectors ($Q^{T}Q=I$), that is, $\lVert q_{i} \rVert  = 1$, $q_{i}^{T}q_{j} = 0$ if $i \neq j$. Moreover, diagonal elements $R_{ii}$ are nonzero ($R$ is nonsingular), and most definitions require $R_{ii} > 0$, this makes $Q$ and $R$ unique.

Before dive into how to calculate QR factorisation, we should know what problem or application we can tackle with or apply for.

* Linear equations
* Generalised linear regression model 
* Singular-value decomposition in the Jacobi-Kogbetliantz approach
* Automatic removal of an object from an image

## Algorithms for QR

1. Gram-Schmidt process
* Complexity is $2mn^{2}$ flops
* Not recommended in practice (sensitive to rounding errors)
2. Modified Gram-Schmidt process
* Complexity is $2mn^{2}$ flops
* Better numerical properties
3. Householder transformations
* Complexity is $2mn^{2}-\frac{2}{3}n^{3}$ flops
* Represents $Q$ as a product of elementary orthogonal matrices
* The most widely used algorithm

### Gram-Schmidt Process

The goal of Gram-Schmidt process is to calculate orthogonal basis $\vec{u_{1}}, \vec{u_{2}}, \cdots, \vec{u_{n}}$ from original basis $\vec{v_{1}}, \vec{v_{2}}, \cdots, \vec{v_{n}}$, and this can also be represented in summation notation:

$$
\vec{u_{k}} = \vec{v_{k}} - \sum_{i=1}^{k-1} \frac{\vec{v_{k}} \cdot \vec{u_{i}}}{\lVert \vec{u_{i}} \rVert^2} \vec{u_{i}}
$$

A full calculation process can be found in this youtube [video](https://www.youtube.com/watch?v=zHbfZWZJTGc&ab_channel=ProfessorDaveExplains) presented by Dave.

#### Matlab code

```matlab
function [Q, R] = gram_schmidt_qr(A)
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    for j = 1:n
        R(1:j-1, j) = Q(:, 1:j-1)' * A(:, j);
        v = A(:, j) - Q(:, 1:j-1) * R(1:j-1, j);
        R(j, j) = norm(v);
        Q(:, j) = v / R(j, j);
    end;
```

#### Python code

```python
class Decomposition:
    def __init__(self):
        pass

    def gram_schmidt_qr(self, A):
        m, n = A.shape
        Q = np.zeros(shape=(m, n))
        R = np.zeros(shape=(n, n))
        for j in range(n):
            v = A[:, j]
            for i in range(j):
                q = Q[:, i]
                R[i, j] = q.dot(v)
                v = v - R[i, j] * q
            Q[:, j] = v / np.linalg.norm(v)
            R[j, j] = np.linalg.norm(v)
        return Q, R
```

### Modified Gram-Schmidt Process

In 1966 John Rice showed by experiments that the two different versions of the Gram–Schmidt orthogonalization, classical (CGS) and modified (MGS) have very different properties when executed in finite precision arithmetic. Only for n = 2 are CGS and MGS numerically equivalent.

Instead of computing the vector $\vec{u_{k}}$ as $\vec{u_{k}} = \vec{v_{k}} - \sum_{i=1}^{k-1} \frac{\vec{v_{k}} \cdot \vec{u_{i}}}{\lVert \vec{u_{i}} \rVert^2} \vec{u_{i}}$, it is computed as:

<div>
    <span>
        \begin{align}
            \vec{u_{k}}^{(1)} &= \vec{v_{k}} - proj_{\vec{u_{1}}} \vec{v_{k}} \\
            \vec{u_{k}}^{(2)} &= \vec{u_{k}}^{(1)} - proj_{\vec{u_{2}}} \vec{u_{k}}^{(1)} \\
            \vdots \\
            \vec{u_{k}}^{(k-1)} &= \vec{u_{k}}^{(k-2)} - proj_{\vec{u_{k-1}}} \vec{u_{k}}^{(k-2)} \\
            \vec{u_{k}} &= \frac{\vec{u_{k}}^{(k-1)}}{\lVert \vec{u_{k}}^{(k-1)} \rVert}
        \end{align}
    </span>
</div>

where $proj_{\vec{u}}\vec{v} = \frac{\vec{v} \cdot \vec{u}}{\lVert \vec{u} \rVert^2} \vec{u}$.

#### Matlab code

```matlab
function [Q, R] = modified_gram_schmidt_qr(A)
    [m, n] = size(A);
    Q = A;
    R = zeros(n);
    for k = 1:n
        R(k, k) = norm(Q(:, k));
        Q(:, k) = Q(:, k) / R(k, k);
        R(k, k+1:n) = Q(:,k)' * Q(:, k+1:n);
        Q(:, k+1:n) = Q(:, k+1:n) - Q(:, k) * R(k, k+1:n);
    end
```

#### Python code

```python
class Decomposition:
    def __init__(self):
        pass

    def modified_gram_schmidt_qr(self, A):
        m, n = A.shape
        Q = np.zeros(shape=(m, n))
        R = np.zeros(shape=(n, n))
        for j in range(0, n):
            R[j, j] = np.sqrt(np.dot(A[:, j], A[:, j]))
            Q[:, j] = A[:, j] / R[j, j]
            for i in range(j+1, n):
                R[j, i] = np.dot(Q[:, j], A[:, i])
                A[:, i] = A[:, i] - R[j, i] * Q[:, j]
        return Q, R
```

### Householder Transformations

Householder transformations are simple orthogonal transformations corresponding to reflection through a plane. Reflection across the plane orthogonal to a unit normal vector $v$ can be expressed in matrix form as

$$
H = I - 2vv^T
$$

In particular, if we take $u=x-s \lVert x \rVert e_{1}$ where $s= \pm 1$ and $v=u/ \lVert u \rVert$ then

$$
Hx = (I - 2 \frac{uu^T}{u^Tu})x = s \lVert x \rVert e_{1}
$$

Let us first verify that this works:

<div>
    <span>
        \begin{align}
            u^Tx &= (x-s \lVert x \rVert e_{1})^T x \\
                 &= \lVert x \rVert^2 - s x_{1} \lVert x \rVert \\
            u^Tu &= (x-s \lVert x \rVert e_{1})^T(x-s \lVert x \rVert e_{1}) \\
                 &= \lVert x \rVert^2 - 2 s x_{1} \lVert x \rVert + \lVert x \rVert^2 \lVert e_1 \rVert^2 \\
                 &= 2 (\lVert x \rVert^2 - s x_1 \lVert x \rVert) \\
                 &= 2 u^T x \\
            Hx   &= x - 2 u \frac{u^T x}{u^T u} = x - u = s \lVert x \rVert e_1
        \end{align}
    </span>
</div>

As a byproduct of this calculation, note that we have

$$
u^Tu = -2 s \lVert x \rVert u_1
$$

where $u_1 = x_1 - s \lVert x \rVert$; and if we define $w = u/u_1$, we have

$$
H = I - 2 \frac{ww^T}{w^Tw} = I + \frac{su_1}{\lVert x \rVert} ww^T = I - \tau ww^T
$$

where $\tau = -s u_1 / \lVert x \rVert$.

#### Matlab code

```matlab
function [Q,R] = householder_qr(A)
    [m, n] = size(A);
    Q = eye(m);
    R = A;
    I = eye(n);

    for j = 1:n-1
        x = R(j:n, j);
        v = -sign(x(1)) * norm(x) * eye(n-j+1, 1) - x;
        if norm(v) > 0
            v = v / norm(v);
            P = I;
            P(j:n, j:n) = P(j:n, j:n) - 2*v*v';
            R = P * R;
            Q = Q * P;
        end
    end
```

#### Python code

```python
class Decomposition:
    """
    https://stackoverflow.com/a/53493770/15048366
    """
    def __init__(self):
        pass

    def householder_vectorised(self, arr):
        v = arr / (arr[0] + np.copysign(np.linalg.norm(arr), arr[0]))
        v[0] = 1
        tau = 2 / (v.T @ v)
        return v, tau

    def householder_qr(self, A):
        m, n = A.shape
        Q = np.identity(m)
        R = A.copy()

        for j in range(0, n):
            v, tau = self.householder_vectorised(R[j:, j, np.newaxis])
            H = np.identity(m)
            H[j:, j:] -= tau * (v @ v.T)
            R = H @ R
            Q = H @ Q

        Q, R = Q[:n].T, np.triu(R[:n])
        for i in range(n):
            if R[i, i] < 0:
                Q[:, i] *= -1
                R[i, :] *= -1
                
        return Q, R
```

if $m > n$:

* Full QR factorisation
![](/assets/img/2021-01-29-matrix-decomposition/full_qr.png)

* Reduced QR factorisation
![](/assets/img/2021-01-29-matrix-decomposition/reduced_qr.png)

# Linear Function

After decomposing matrix $A$, you can write a function in python to solve a system

$$
Ax = b
$$

using LU decomposition and QR decomposition. Your function should take $A$ and $b$ as input and return $x$.

Function should include the following:

* Check that $A$ is not a singular matrix, that is, $A$ is invertible.
* Invert $A$ using different decomposition methods and solve
* Return x

```python
class Decomposition:
    
    def plu(self, A):
        n = A.shape[0]
        P = np.identity(n)
        L = np.identity(n)
        U = A.copy()
        
        PF = np.identity(n)
        LF = np.zeros(shape=(n, n))
        
        # Loop over rows
        for i in range(n-1):
            index = np.argmax(abs(U[i:, i]))
            index = index + i
            if index != i:
                P = np.identity(n)
                P[[index, i], i:n] = P[[i, index], i:n]
                U[[index, i], i:n] = U[[i, index], i:n] 
                PF = np.dot(P, PF)
                LF = np.dot(P, LF)
            L = np.identity(n)
            for j in range(i+1, n):
                L[j, i]  = -(U[j, i] / U[i, i])
                LF[j, i] =  (U[j, i] / U[i, i])
            U = np.dot(L, U)
        np.fill_diagonal(LF, 1)
        return PF, LF, U
    
    def gram_schmidt_qr(self, A):
        m, n = A.shape
        Q = np.zeros(shape=(m, n), dtype='float64')
        R = np.zeros(shape=(n, n), dtype='float64')
        for j in range(n):
            v = A[:, j]
            for i in range(j):
                q = Q[:, i]
                R[i, j] = q.dot(v)
                v = v - R[i, j] * q
            Q[:, j] = v / np.linalg.norm(v)
            R[j, j] = np.linalg.norm(v)
        return Q, R
    
    def modified_gram_schmidt_qr(self, A):
        n = A.shape[1]
        Q = np.array(A, dtype='float64')
        R = np.zeros((n, n), dtype='float64')
        for k in range(n):
            a_k = Q[:, k]
            R[k,k] = np.linalg.norm(a_k)
            a_k /= R[k, k]
            for i in range(k+1, n):
                a_i = Q[:, i]
                R[k,i] = np.transpose(a_k) @ a_i
                a_i -= R[k, i] * a_k
        return Q, R
    
    def householder_vectorised(self, arr):
        v = arr / (arr[0] + np.copysign(np.linalg.norm(arr), arr[0]))
        v[0] = 1
        tau = 2 / (v.T @ v)
        return v, tau

    def householder_qr(self, A):
        m, n = A.shape
        Q = np.identity(m)
        R = A.copy()

        for j in range(0, n):
            v, tau = self.householder_vectorised(R[j:, j, np.newaxis])
            H = np.identity(m)
            H[j:, j:] -= tau * (v @ v.T)
            R = H @ R
            Q = H @ Q

        Q, R = Q[:n].T, np.triu(R[:n])
        for i in range(n):
            if R[i, i] < 0:
                Q[:, i] *= -1
                R[i, :] *= -1
                
        return Q, R

def linear_function_solver(A, b, method="LU"):
    det = ChioDeterminants().calculate(A)
    factoriser = Decomposition()
    if det == 0:
        print("Matrix is singular!")
        return
    if method == "LU":
        P, L, U = factoriser.plu(A)
        z_1 = P.T @ b
        z_2 = np.linalg.inv(L) @ z_1
        x = np.linalg.inv(U) @ z_2
        return x
    elif method == "CGS":
        Q, R = factoriser.gram_schmidt_qr(A)
        x = np.linalg.inv(R) @ Q.T @ b
        return x
    elif method == "MGS":
        Q, R = factoriser.modified_gram_schmidt_qr(A)
        x = np.linalg.inv(R) @ Q.T @ b
        return x
    elif method == "HHT":
        Q, R = factoriser.householder_qr(A)
        x = np.linalg.inv(R) @ Q.T @ b
        return x
```

Let's check on four different approachs.

```python
A = np.array([
    [8, 6, 4, 1], 
    [1, 4, 5, 1], 
    [7, 4, 2, 5], 
    [1, 4, 2, 6]])
b = np.array([20, 12, 23, 19])

print("NP:  ", np.linalg.solve(A, b))
print("LU:  ", linear_function_solver(A, b, method="LU"))
print("CGS: ", linear_function_solver(A, b, method="CGS"))
print("MGS: ", linear_function_solver(A, b, method="MGS"))
print("HHT: ", linear_function_solver(A, b, method="HHT"))
```

---

```python
NP:   [1. 1. 1. 2.]
LU:   [1. 1. 1. 2.]
CGS:  [1. 1. 1. 2.]
MGS:  [1. 1. 1. 2.]
HHT:  [1. 1. 1. 2.]
```

# Conclusion

In this article, I implement different matrix decomposition methods, named LU decomposition and QR decomposition (Gram-Schmidt process, Modified Gram-Schmidt process, Householder transformations). In the future, I may apply matrix decomposition algorithm to neural networks. I hope it will be much more efficient than the regularisers methods.

## References

1. https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/04LinearAlgebra/lup/
2. https://math.unm.edu/~loring/links/linear_s08/LU.pdf
3. https://johnfoster.pge.utexas.edu/numerical-methods-book/
4. https://web.cs.ucdavis.edu/~bai/publications/andersonbaidongarra92.pdf
5. https://deepai.org/machine-learning-glossary-and-terms/qr-decomposition
6. https://en.wikipedia.org/wiki/Matrix_decomposition
7. http://homepages.math.uic.edu/~jan/mcs507f13/
8. https://www.cis.upenn.edu/~cis610/Gram-Schmidt-Bjorck.pdf
9. https://wikivisually.com/wiki/Gram%E2%80%93Schmidt_process
10. https://rpubs.com/aaronsc32/qr-decomposition-householder
11. https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
12. https://homel.vsb.cz/~dom033/predmety/parisLA/05_orthogonalization.pdf
13. https://core.ac.uk/download/pdf/82066579.pdf
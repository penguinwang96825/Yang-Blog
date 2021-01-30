---
layout: post
title: Laplace Expansion and Chiò Condensation
subtitle: An Algorithm that has Almost not been Passed Down
cover-img: /assets/img/2021-01-28-laplace-expansion-and-chio-algorithm/call-me-lambh.jpg
thumbnail-img: /assets/img/2021-01-28-laplace-expansion-and-chio-algorithm/course-image.png
readtime: true
show-avatar: false
tags: [Python, Mathematics, LinearAlgebra]
comments: true
---

Determinants are mathematical objects which have applications in engineering mathematics. For example, they can be used in the solution of simultaneous equations, and to evaluate vector products. Determinants can also be used to see if a system of $n$ linear equations in $n$ variables has a unique solution. There are several ways to calculate determinant, however, today I'm going to introduce another way of computing determinants: Chiò Identity.

# Introduction

In 1853 Felice (Félix) Chiò (1813–1871) published his short "Mémoire sur les fonctions connues sous le noms De Résultantes Ou De Déterminans". In this article, I first give a way of evaluating determinants by Laplace Expansion, and explicitly comparing Chiò Identity to this.

# Laplace Expansion Method

In linear algebra, the Laplace expansion, named after Pierre-Simon Laplace, also called cofactor expansion, is an expression for the determinant $B$ of an $n \times n$ matrix $A$ that is a weighted sum of the determinants of $n$ sub-matrices (or minors) of $B$, each of size $(n − 1) \times (n − 1)$. The $i$, $j$ cofactor of the matrix $A$ is the scalar $C_{ij}$ defined by $C_{ij}=(-1)^{i+j}M_{ij}$, where $M_{ij}$ is the $i$, $j$ minor of $A$, that is, the determinant of the $(n − 1) \times (n − 1)$ matrix that results from deleting the $i$-th row and the $j$-th column of $A$.

```python
class LaplaceDeterminants:
    def __init__(self):
        pass
    
    def minor_matrix(self, A, i, j):
        # Delete i-th row
        sub_A = np.delete(A, i, 0)
        # Delete j-th column
        sub_A = np.delete(sub_A, j, 1)
        return sub_A
    
    def calculate(self, A):
        n, m = A.shape
        if not n == m: 
            raise Exception("Must be a square matrix!")
        if n == 2:
            return A[0][0]*A[1][1] - A[1][0]*A[0][1]
        det = 0
        for i in range(n):
            M = self.minor_matrix(A, 0, i)
            det += (-1)**i * A[0][i] * self.calculate(M)
        return det
```

# Chiò Condensation Method

The statement of Chiò Condensation is: let $A=(a_{ij})$ be an $n \times n$ matrix, and let $a_{11} \neq 0$. Replace each element $a_{ij}$ in the $(n-1) \times (n-1)$ sub-matrix, let's called it $D$, of $A$ obtained by deleting the $i$th row and $j$th column of $A$ by:

<div>
    <span>
        \[
           D = 
          \left[ {\begin{array}{cc}
           a_{ij} & a_{in} \\
           a_{nj} & a_{nn} \\
          \end{array} } \right]
        \]
    </span>
</div>

Then $det(A)= \frac{1}{a_{nn}^{n-2}} \cdot det(D)$.

```python
class ChioDeterminants:
    def __init__(self):
        pass
    
    def calculate(self, A):
        n, m = A.shape
        if not n == m: 
            raise Exception("Must be a square matrix!")
        if n == 2:
            return A[0][0]*A[1][1] - A[1][0]*A[0][1]
        if A[-1][-1] == 0:
            for i in range(n):
                if A[0][i] != 0:
                    A[:, [i, n-1]] = A[:, [n-1, i]]
                    A[[0, n-1], :] = A[[n-1, 0], :]
                    break
                else:
                    return 0
        D = np.zeros(shape=(n-1, n-1))
        for i in range(n-1):
            for j in range(n-1):
                D[i][j] = A[i][j]*A[-1, -1] - A[-1][j]*A[i][-1]
        det = (1/A[-1][-1]**(n-2)) * self.calculate(D)
        return det
```

# Performance

```python
def test_laplace(n_samples=50000):
    algo = LaplaceDeterminants()
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = algo.calculate(A)

def test_chio(n_samples=50000):
    algo = ChioDeterminants()
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = algo.calculate(A)
```

![](/assets/img/2021-01-28-laplace-expansion-and-chio-algorithm/perf1.png)

What if we also compare Chiò Condensation Method to `numpy` and `scipy`? They both computed determinants via LU factorization, relying on BLAS and [LAPACK](http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html) to provide efficient low level implementations of standard linear algebra algorithms.

```python
def test_numpy(n_samples=50000):
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = np.linalg.det(A)

def test_scipy(n_samples=50000):
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = linalg.det(A)
```

![](/assets/img/2021-01-28-laplace-expansion-and-chio-algorithm/perf2.png)

# Conclusion

Clearly, Chiò Condensation Method is much quicker than Laplace Expansion Method by minors, which yeilds complexity computation of $O(n!)$. As an alternative method for hand-calculating determinants, therefore, Chiò's method is quite effective. For numerical computations of large determinants on a computer, however, Chiò's method is not so efficient as other methods such as, for example, Gaussian elimination, because of certain difficulties with round-off errors. In addition, Chiò's method requires approximately $\frac{2}{3}n^3$ multiplications, whereas Gaussian elimination requires approximately $\frac{1}{3}n^3$. 

## References

1. https://www.sciencedirect.com/science/article/pii/S0024379514002249
2. https://www.codeformech.com/determinant-linear-algebra-using-python/
3. https://en.wikipedia.org/wiki/Laplace_expansion
4. https://stackoverflow.com/questions/16636858/complexity-computation-of-a-determinant-recursive-algorithm
5. Fuller, L. E., & Logan, J. D. On the Evaluation of Determinants by Chiò Method, 1975
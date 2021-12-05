K-SVD implementation for word embedding
---------------------------------------

#### K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation
https://sites.fas.harvard.edu/~cs278/papers/ksvd.pdf


Objective Function

<img width="483" alt="Screen Shot 2021-11-26 at 12 38 25 PM" src="https://user-images.githubusercontent.com/22663880/143620542-623fbee7-1bce-4186-85e6-fca6c553e94b.png">

Pseudo Code:

1. Sparse Coding Stage

 <img width="513" alt="Screen Shot 2021-11-26 at 12 39 48 PM" src="https://user-images.githubusercontent.com/22663880/143620734-add15d72-ca61-4224-b81a-5aba04469f6a.png">
 
 - Fix D and find best coefficient matrix X

2. Find new column d_k and new values for its coefficients (kth row of X)

<img width="488" alt="Screen Shot 2021-11-26 at 12 40 02 PM" src="https://user-images.githubusercontent.com/22663880/143620946-d3b777e7-9c90-4f34-b76d-2e9cbe1818fc.png">

- Here The matrix E_k stands for the error for all the N examples when the kth atom is removed

<img width="349" alt="Screen Shot 2021-11-26 at 12 40 23 PM" src="https://user-images.githubusercontent.com/22663880/143621370-f44c5c4a-0475-4164-91f2-686790585f39.png">

- Define W_k as the group of indices pointing to examples {y_i} that use the atom  d_k, i.e., those where x_k_T(i) is non-zero
- Define Omega_k as a matrix of size N * |W_k| with ones on the (W_k(i), i)th entries and zeros elsewhere
- <img width="381" alt="Screen Shot 2021-11-26 at 12 40 45 PM" src="https://user-images.githubusercontent.com/22663880/143621577-5ba0f4b6-227e-44f7-ad76-7de3a4bcfbb3.png">

    * SVD(E_k_R) = USVT
    * First column of U is solution for d_k
    * First column of V multiplied by S(1,1) is x_R_k


#### Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit

https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

Batch OMP is designed for sparse coding large sets of signals
1. Uses Batch OMP for sparse coding method
2. Update dictionary and weights in an optimized way
![lagrida_latex_editor-2](https://user-images.githubusercontent.com/22663880/144762383-0b9cf066-25b0-4ce9-8c0d-42aad73cf654.png)


Version of https://github.com/nel215/ksvd

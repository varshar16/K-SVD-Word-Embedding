# K-SVD application using word embeddings 


We want to represent each word vector as a sparse linear combination of atom vectors.


**Input:**
1. `matrix Y` containing *word embeddings of size* `V * N` where `V` is the *number of vectors* and `N` is the *dimension of the semantic space*. 
2. `tol` denotes target sparsity
3. `k` is number of iterations

**Outputs:**
1. `Matrix D` containing *atom vectors corresponding to topics*. Size is `N * K` where `K` is the *number of topics determined by the user*.
   - Since each atom has the same dimension as word vectors, we can compare these atoms to words (e.g. with cosine similarity) and understand what topic each atom represents.
2. `Matrix X` of size `K * V` where *each column indicates how to reconstruct a word as a linear combination of atom vectors*. 
   - Which atom vector to use to reconstruct a word and in what amount. 
   - Can see how much each word contributed to a topic.

**Pseudo Code for Aprroximate K-SVD**

![lagrida_latex_editor-9](https://user-images.githubusercontent.com/22663880/144764448-ebd57bfe-d6fa-4ae7-8a3f-8ab274ba3599.png)<br>
![lagrida_latex_editor-10](https://user-images.githubusercontent.com/22663880/144764501-5e859db9-de20-4145-8274-4838e33415b2.png)<br>
&nbsp;&nbsp;&nbsp;&nbsp;![lagrida_latex_editor-7](https://user-images.githubusercontent.com/22663880/144764271-d543044f-7196-4b35-b961-96525af7c4c3.png)<br>
&nbsp;&nbsp;&nbsp;&nbsp;![lagrida_latex_editor-5](https://user-images.githubusercontent.com/22663880/144764004-04a81463-562f-4168-a7e6-36c4b8f81857.png)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![lagrida_latex_editor-4](https://user-images.githubusercontent.com/22663880/144763589-d7dfb681-2e25-4a5b-97eb-d0874ed6b75f.png) <br>
&nbsp;&nbsp;&nbsp;&nbsp;![lagrida_latex_editor-6](https://user-images.githubusercontent.com/22663880/144764037-5a079f4f-3508-448c-bd46-e3a53b344986.png)<br>
![lagrida_latex_editor-6](https://user-images.githubusercontent.com/22663880/144764037-5a079f4f-3508-448c-bd46-e3a53b344986.png)<br>


## Example Usage

```
# Default values for k=10, tol=1e-6
V = 100
N = 10
Y = np.random.rand(V,N)
ksvd = ApproxKSVD(V, N)
D, X = ksvd.fit(Y)
```

Note: This implementation is a version of https://github.com/nel215/ksvd

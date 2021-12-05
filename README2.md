# K-SVD application using word embeddings 


We want to represent each word vector as a sparse linear combination of atom vectors.


**Input:**
Takes in a `matrix Y` containing *word embeddings of size* `V * N` where `V` is the *number of vectors* and `N` is the *dimension of the semantic space*.


**Outputs:**
1. `Matrix D` containing *atom vectors corresponding to topics*. Size is `N * K` where `K` is the *number of topics determined by the user*.
   - Since each atom has the same dimension as word vectors, we can compare these atoms to words (e.g. with cosine similarity) and understand what topic each atom represents.
2. `Matrix X` of size `K * V` where *each column indicates how to reconstruct a word as a linear combination of atom vectors*. 
   - Which atom vector to use to reconstruct a word and in what amount. 
   - Can see how much each word contributed to a topic. 


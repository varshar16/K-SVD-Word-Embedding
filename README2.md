# K-SVD application using word embeddings 
<p>We want to represent each word vector as a sparse linear combination of atom vectors. <br> <br>
  <b>Input:</b> Takes in a matrix <b>Y</b> containing <u>word embeddings of size V * N </u> <br>
where V is the number of vectors and N is the dimension of the semantic space. <br>
<b>Outputs:</b>
Matrix D containing atom vectors corresponding to topics. Size is N * K where K is the number of topics determined by the user.
Since each atom has the same dimension as word vectors, we can compare these atoms to words (e.g. with cosine similarity) and understand what topic each atom represents. 
Matrix X of size K * V where each column indicates how to reconstruct a word as a linear combination of atom vectors. 
Which atom vector to use to reconstruct a word and in what amount. 
Can see how much each word contributed to a topic. 
</p>

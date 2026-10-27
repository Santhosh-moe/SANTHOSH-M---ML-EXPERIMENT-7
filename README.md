### DATE : 06/10/2026






### AIM: 





Reduce 4D Iris data to 2 principal components and visualize species separation.






### ALGORITHM : 







PCA via SVD: compute components that capture maximal variance.









### THEORY:







PCA finds orthogonal directions of maximal variance (eigenvectors of covariance matrix).






## INPUT : 



```
# EXPERIMENT 6
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
iris = load_iris()
X = iris.data; y = iris.target
pca = PCA(n_components=2)
Z = pca.fit_transform(X)
print('Explained variance ratio:',
      np.round(pca.explained_variance_ratio_,4))
plt.scatter(Z[:,0], Z[:,1], c=y, cmap='viridis'); plt.xlabel('PC1');
plt.ylabel('PC2'); plt.title('PCA of Iris'); plt.show()


```





## OUTPUT:


<img width="578" height="455" alt="image" src="https://github.com/user-attachments/assets/ca7e8b66-1413-4f00-a399-b92794c78ece" />



### RESULT: 






Students see PC1 captures most variance; PCA useful for visualization and 
dimensionality reduction.

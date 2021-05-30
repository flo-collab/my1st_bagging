from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from mglearn.plot_interactive_tree import plot_tree_partition
from mglearn.plot_2d_separator import plot_2d_separator
from mglearn.tools import discrete_scatter
import matplotlib.pyplot as plt

# On crée un jeu de données d'exemple: 
X, y = make_moons(n_samples=100, noise=0.25)
# On le separe en 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
 
# On crée notre model. Par default le classifieur de base est un arbre de desicion
# On précise le nombre de classifieurs individuels utilisés ici 5
bagging = BaggingClassifier(n_estimators=5)
bagging.fit(X_train, y_train)

# On parametre nos graphs : 
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), bagging.estimators_)):
    ax.set_title("Tree {}".format(i))
    plot_tree_partition(X_train, y_train, tree, ax=ax)
plot_2d_separator(bagging, X_train, fill=True, ax=axes[-1, -1],
                                    alpha=.4)
axes[-1, -1].set_title("Bagging")
discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()
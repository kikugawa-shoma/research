import matplotlib.pyplot as plt
from tools.mapalign.embed import DiffusionMapEmbedding
import numpy as np
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.cluster import KMeans

def compare_methods(X, title, cluster=None):
    f, axarr = plt.subplots(2, 7, sharex=True, sharey=True, figsize=(15, 5))
    for idx, t in enumerate([0, 0.1, 1, 10, 100, 250, 1000]):
        de = DiffusionMapEmbedding(alpha=0.5, diffusion_time=t, affinity='markov',
                                   n_components=10).fit_transform(X.copy())
        ed = (de - de[0, :])
        ed = np.sqrt(np.sum(ed * ed , axis=1))
        ed = ed/max(ed)
        if cluster is not None:
            ed = KMeans(n_clusters=cluster).fit(de).labels_
        plt.axes(axarr[0][idx])
        if cluster is not None: 
            plt.scatter(X[:, 0], X[:, 1], c=ed, cmap=plt.cm.Set1, linewidths=0)
        else:
            plt.scatter(X[:, 0], X[:, 1], c=ed, cmap=plt.cm.Spectral, linewidths=0)
        plt.axis('tight')
        if cluster is None:
            plt.colorbar()
        plt.title('t={:g}'.format(t))
    for idx, c in enumerate([2, 3, 4, 5, 10, 20, 50]):
        se = SpectralEmbedding(n_components=c).fit_transform(X.copy())
        ed = (se - se[0, :])
        ed = np.sqrt(np.sum(ed * ed , axis=1))
        ed = ed/max(ed)
        if cluster is not None:
            ed = KMeans(n_clusters=cluster).fit(ed[:, None]).labels_
        plt.axes(axarr[1][idx])
        if cluster is not None: 
            plt.scatter(X[:, 0], X[:, 1], c=ed, cmap=plt.cm.Set1, linewidths=0)
        else:
            plt.scatter(X[:, 0], X[:, 1], c=ed, cmap=plt.cm.Spectral, linewidths=0)
        plt.axis('tight')
        if cluster is None:
            plt.colorbar()
        plt.title('num_c=%d' % (c))
    ph = plt.suptitle(title)
    plt.show()

n=2000
t=np.power(np.sort(np.random.rand(n)), .7)*10
al=.15;bet=.5;
x1=bet * np.exp(al * t) * np.cos(t) + 0.1 * np.random.randn(n)
y1=bet * np.exp(al * t) * np.sin(t) + 0.1 * np.random.randn(n)
X = np.hstack((x1[:, None], y1[:, None]))
n=2000
t=np.power(np.sort(np.random.rand(n)), .7)*10
al=.15;bet=.5;
x1=bet * np.exp(al * t) * np.cos(t) + 0.1 * np.random.randn(n)
y1=bet * np.exp(al * t) * np.sin(t) + 0.1 * np.random.randn(n)
X = np.hstack((x1[:, None], y1[:, None]))

plt.scatter(x1, y1, c=t, cmap=plt.cm.Spectral, linewidths=0)
ph = plt.plot(x1[0], y1[0], 'ko')
plt.show()

compare_methods(X, 'Noisy spiral')
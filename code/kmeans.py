#!usr/bin/env python3

"""Find K topics from Problem or Method."""

from os.path import join

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


def get_text(transactions, value2dim, dimension="Method"):
    """Get only 1 dimension value from transactions and return a list of
    dimension values and their corresponding paper."""
    text = []
    paper_list = []
    for paper in transactions:
        dim_text = []
        for value in transactions[paper]:
            if value2dim[value] == dimension:
                dim_text.append(value)
        if dim_text != []:
            paper_list.append(paper)
            text.append(" ".join(dim_text))
    return text, paper_list


def kmeans_model(text, K_clusters, paper_list, numfeatures=1000,
                 n_components=30):
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.5,
                                 ngram_range=(1, 3),
                                 max_features=numfeatures,
                                 stop_words="english",
                                 use_idf=True)
    X = vectorizer.fit_transform(text)
    pca = PCA(n_components)
    X_r = pca.fit(X.A).transform(X.A)
    kmeans = KMeans(n_clusters=K_clusters, random_state=0).fit(X_r)
    labels = kmeans.labels_
    label2paper = {}
    for label, paper in zip(labels, paper_list):
        if label not in label2paper:
            label2paper[label] = [paper]
        else:
            label2paper[label].append(paper)
    score = silhouette_score(X, labels, metric="cosine")
    print("The mean Silhouette Coefficient is {:.2f}\n".format(score))
    return X, labels, label2paper


def plot_cluster(X, K_clusters, labels, path2output, dimension="Method"):
    pca = PCA(n_components=2)
    X_r = pca.fit(X.A).transform(X.A)
    palette = sns.color_palette("husl", K_clusters)
    colors_list = list(map(lambda x: palette[x], labels))
    sns.set_style("ticks", {"xtick.direction": "in",
                            "ytick.direction": "in"})
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_r[:, 0], X_r[:, 1], c=colors_list, s=16)
    plt.setp(ax, xlabel="Principle component 1", ylabel="Principle component 2")
    plt.savefig(join(path2output, "KMeans_{}.pdf".format(dimension)))
    plt.close(fig)


def make_wordcloud(transactions, label2paper, value2dim, path2output,
                   dimension="Method"):
    for label in label2paper:
        content = []
        for paper in label2paper[label]:
            for value in transactions[paper]:
                if value2dim[value] == dimension:
                    content.append(value)
        wc = WordCloud(background_color="white", max_words=20,
                       width=1200, height=1000, random_state=0)
        wc.generate(" ".join(content))
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        fig.tight_layout()
        figname = "WordCloud_Topic{:d}.pdf".format(label)
        plt.savefig(join(path2output, figname))
        plt.close(fig)


def run_kmeans(transactions, value2dim, K_clusters, path2output):
    text, paper_list = get_text(transactions, value2dim)
    X, labels, label2paper = kmeans_model(text, K_clusters, paper_list)
    plot_cluster(X, K_clusters, labels, path2output)
    make_wordcloud(transactions, label2paper, value2dim, path2output)
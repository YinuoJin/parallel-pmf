import os
import subprocess
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import plot as pmf_plot

class PMF(object):
    """
    Python wrapper for the C++ program Parallel Probablistic Matrix Factorizattion (PMF)
    """
    def __init__(self, indir, outdir, task='train', **kwargs):
        """
        Parameters
        ----------
        indir : str
            Parent directory for input datasets

        outdir : str
            Output directory to save model parameters & plots

        task : str
            Task for the model to perform
            task = 'train' - perform training to learn latent model parameters
            task = 'recommend' - perform recommendation based on learnt parameters

        parallel : bool
            Whether to perform multi-thread parallelization

        thread : int
            Number of theads during parallelized training

        theta_std : float
            Standard deviation of the prior distribution for theta vectors

        beta_std: float
            Standard deviation of the prior distribution for beta vectors
        """
        pwd = os.path.join(os.path.dirname(__file__), 'bin')
        exec_name = 'main.tsk'
        self.bin_exec = os.path.join(pwd, exec_name)
        self.indir = indir
        self.data_path = os.path.join(indir, 'expr.csv')
        self.mapper_path = os.path.join(indir, 'genes.csv')
        self.outdir = outdir
        self.task = task

        # model parameters
        self.theta = pd.DataFrame()  # dimension: spot_id x k-dimensional theta latent attribute
        self.beta = pd.DataFrame() # dimension: item_id x k-dimensional beta latent attribute
        self.beta_embedded = pd.DataFrame() # dimension: item_id x 3

        self.loss = pd.DataFrame()

        # dataset information
        self.spots = set()
        self.items = set()
        self.genres = set()
        # mapping information betweeen item, item title & genre
        self.item_title = {}
        self.item_genre = {}
        self.title_item = {}
        self.title_genre = {}
        self.genre_items = {}

        assert os.path.exists(self.bin_exec), "Invalid binary executable"
        assert os.path.exists(self.data_path), "Input data file {} doesn't exist".format(self.data_path)
        assert os.path.exists(self.mapper_path), "Input mapping file {} doesn't exist".format(self.mapper_path)

        self._initialize(kwargs)

    def _initialize(self, kwargs):
        self.default_params = {
            'thread': 8,
            'gamma': 0.01,
            'lambda_theta': 0.5,
            'lambda_beta': 0.5,
            'eta_theta': 0.1,
            'eta_beta': 5.0,
        }
        print('Initializing model parameters...')

        self.args = {}
        for key in self.default_params.keys():
            if key in kwargs.keys():
                self.args[key] = kwargs[key]
            else:
                self.args[key] = self.default_params[key]

    def learn(self, k=5, n_epochs=200, train_test_split=0.75):
        """
        Perform model training to learn model parameters given the dataset
        It calls the C++ program to perform training and save the training
        results to output directory

        Parameters
        ----------
        k : int
            Vector length for each theta & beta latent vector

        n_epochs : int
            Number of iterations for training

        train_test_split : float
            Ratio for random splitting the dataset into training & test sets
        """

        cmd = [self.bin_exec,
               "--task {}".format(self.task),
               "-i {}".format(self.data_path),
               "-m {}".format(self.mapper_path),
               #"-o {}".format(self.outdir),
               "-k {}".format(k),
               "-n {}".format(n_epochs),
               "-r {}".format(train_test_split),
               "--thread {}".format(self.args['thread']),
               "--gamma {}".format(self.args['gamma']),
               "--lambda_theta {}".format(self.args['lambda_theta']),
               "--eta_theta {}".format(self.args['eta_theta']),
               "--lambda_beta {}".format(self.args['lambda_beta']),
               "--eta_beta {}".format(self.args['eta_beta']),
        ]
        
        
        #cmd = [self.bin_exec, "-h"]
        
        print('Training model...')
        res = subprocess.getoutput(' '.join(cmd))
        print(res)

    def load(self, indir):
        """Load learnt model parameters & item mapping info. from file"""
        print('Loading previously learnt parameters into model...')
        theta_file = os.path.join(indir, "theta.csv")
        beta_file = os.path.join(indir, "beta.csv")
        loss_file = os.path.join(indir, "loss.csv")

        assert os.path.exists(loss_file), \
            "Loglikelihood hasn't been calculated, please train the model first"

        assert os.path.exists(theta_file) and os.path.exists(beta_file), \
            "Latent vector theta & beta hasn't been learnt, please train the model first"

        self.loss = pd.read_csv(loss_file)
        self.theta = self._load_model(theta_file)
        self.beta = self._load_model(beta_file)
        self.spots = set(self.theta.index)
        self.items = set(self.beta.index)
        self._load_mapper()  # Load item - title - genre maps

    def _verify_load_status(self):
        if len(self.theta) == 0 or len(self.beta) == 0:
            self.load(self.outdir)

    def _load_model(self, file):
        df = pd.read_csv(file)
        return self._process_vectors(df)

    def _load_mapper(self):
        assert os.path.exists(self.mapper_path), \
            "Item mapping file doesn't exist"
        df_mapper = pd.read_csv(self.mapper_path)

        first_genre = df_mapper['itemAttributes'].apply(lambda x: x.strip().split('|')[0])
        df_mapper['first_genre'] = first_genre
        self.genres = set(np.unique(first_genre))
        df_reidx1 = df_mapper.set_index('itemId')
        df_reidx2 = df_mapper.set_index('itemName')

        self.item_title = df_reidx1.to_dict()['itemName']
        self.item_genre = df_reidx1.to_dict()['itemAttributes']
        item_first_genre = df_reidx1.to_dict()['first_genre']
        self.title_item = df_reidx2.to_dict()['itemId']
        self.title_genre = df_reidx2.to_dict()['itemAttributes']

        for item, genre in item_first_genre.items():
            if genre in self.genre_items.keys():
                self.genre_items[genre].add(item)
            else:
                self.genre_items[genre] = {item}

    def _process_vectors(self, df):
        df['vector'] = df['vector'].apply(lambda x: [float(i) for i in x.split()])
        vals = np.array(df['vector'].values.tolist())
        cols = ['attr_' + str(i) for i in range(1, vals.shape[1] + 1)]
        df_processed = pd.DataFrame(data=vals, index=df['id'], columns=cols)

        return df_processed

    def _predict(self, spot_id):
        """Predict the preference of spot_id to all items"""
        theta_i = self.theta.loc[spot_id]
        pred = theta_i.dot(self.beta.T)

        return pred

    def _predict_spot(self, item_id):
        beta_i = self.beta.loc[item_id]
        pred = beta_i.dot(self.theta.T)

        return pred

    def _recommend_item_to_spot(self, item_id, N=10):
        self._verify_load_status()
        preds = self._predict_spot(item_id)
        rec_spots = preds.sort_values(ascending=False)
        rec_use_ids = rec_spots.index[:N].to_series()

        return rec_use_ids

    def recommend_spot(self, spot_id, N=10, verbose=1):
        """Recommend top N items for given spot"""
        self._verify_load_status()
        assert spot_id in self.spots, \
            "Spot id {} doesn't exist in the dataset".format(spot_id)
        if verbose:
            print("Top {0} recommended genes for spot {1}:".format(N, spot_id))

        preds = self._predict(spot_id)
        rec_items = preds.sort_values(ascending=False)
        rec_items = rec_items.index[:N].to_series()
        rec_titles = rec_items.map(self.item_title)
        df_rec = self._refactor_rec(rec_titles)

        return df_rec

    def recommend_items(self, item, N=10, verbose=1):
        """Recommend top N most similar items for given item"""
        self._verify_load_status()
        if isinstance(item, str):
            try:
                item_id = self.title_item[item]
            except KeyError:
                print("Item {} doesn't exist in the dataset".format(item))
        else:
            item_id = item
        assert item_id in self.items, \
            "Item id {0} doesn't exist in the dataset".format(item_id)
        if verbose:
             print("Top {0} recommended genes if you also like {1}:".format(N, self.item_title[item_id]))

        rec_items = self._get_similar_items(item_id, N)
        rec_titles = rec_items.map(self.item_title)
        df_rec = self._refactor_rec(rec_titles)

        return df_rec

    def recommend_joint(self, spot_id, iter=2, N=3, verbose=1):
        """Iterative recommend spots and items for each other"""
        assert spot_id in self.spots, \
            "Spot {} doesn't exist in the dataset".format(spot_id)
        if verbose:
            print("Iteratively recommending spots and items for {} periods...".format(iter))

        rec_spots = {spot_id}
        rec_items = set()
        curr_itr = 0

        while curr_itr < iter:
            for spot in rec_spots:  # recommend spot -> item
                curr_items = self.recommend_spot(spot, N=N, verbose=0).index
                rec_items = rec_items.union(set(curr_items))

            for item in rec_items:  # recommend item -> spot
                curr_spots = self._recommend_item_to_spot(item, N=N)
                rec_spots = rec_spots.union(set(curr_spots))

            curr_itr += 1

        rec_spots = list(rec_spots)
        rec_items = pd.Series(list(rec_items))
        rec_titles = rec_items.map(self.item_title)
        df_rec_items = self._refactor_rec(rec_titles, idx=rec_items)

        return rec_spots, df_rec_items

    def recommend_genre(self, genre, N=10, verbose=1):
        """Recommend top N items from a given genre"""
        self._verify_load_status()
        assert genre in self.genres, \
            "Genre {} doesen't exist in the dataset".format(genre)
        if verbose:
            print("Top {0} recommended genes for genre {1}:".format(N, genre))

        candidate_item_list = list(self.genre_items[genre])
        sample_item_id = np.random.choice(candidate_item_list)
        rec_items = self._get_similar_items(sample_item_id, N)
        rec_titles = rec_items.map(self.item_title)
        df_rec = self._refactor_rec(rec_titles)

        return df_rec

    def _get_similar_items(self, item_id, N):
        beta_j = self.beta.loc[item_id]
        similarity = beta_j.dot(self.beta.T)
        items = similarity.sort_values(ascending=False).index[1:N+1].to_series() # skip index[0] to avoid self
        return items

    def _refactor_rec(self, rec, idx=None):
        genres = rec.map(self.title_genre)
        idx = rec.index if idx is None else idx
        df_rec = pd.DataFrame(zip(rec, genres), index=idx, columns=['itemName', 'itemAttributes'])

        return df_rec

    def _find_optimal_k(self, df):
        score = []
        cluster_labels = []

        for i, k in enumerate(range(3, 10)):
            curr_labels = GaussianMixture(n_components=k).fit(df).predict(df)
            score.append(silhouette_score(df, curr_labels))
            if np.argmax(score) == i:
                cluster_labels = curr_labels

        idx = np.argmax(score)

        return np.argmax(score) + 3, cluster_labels

    def tsne(self):
        """Perform dimension reduction on items to 3 attributes from k-dimension vector space"""
        if self.theta.shape[1] == 3:
            print("The model only inferred k = 3, no need to perform dimension reduction")
            self.beta_embedded = self.beta.copy()
        else:
            print('Performing dimension reduction with t-SNE...')
            self.beta_embedded = TSNE(n_components=3, n_jobs=4).fit_transform(self.beta)
            self.beta_embedded = pd.DataFrame(self.beta_embedded,
                                              index=self.beta.index,
                                              columns=['attr_1', 'attr_2', 'attr_3'])

        return self.beta_embedded

    def clustering(self):
        """Perform EM-clustering on the items given trained beta vectors"""
        print('Performing EM clustering on items...')
        n_clusters, labels = self._find_optimal_k(self.beta_embedded)
        print("Detected {} clusters".format(n_clusters))
        self.beta_embedded['cluster'] = labels

    def display_tsne(self, interactive=True):
        self._verify_load_status()
        if len(self.beta_embedded) == 0:
            self.tsne()
        if 'cluster' not in self.beta_embedded.columns:
            self.clustering()
        if interactive:
            pmf_plot.tsne_interactive(self.beta_embedded)
        else:
            pmf_plot.tsne(self.beta_embedded)

    def display_loss(self):
        """Show loss vs. epoch"""
        self._verify_load_status()
        x = np.arange(self.loss.shape[0]) * 10 + 10
        self.loss['Epoch'] = x
        pmf_plot.loss(self.loss, outdir=self.outdir)

    def display_spot(self, spot_id, N=10, show_title=False, interactive=True):
        self._verify_load_status()
        if len(self.beta_embedded) == 0:
            self.tsne()
        print("Spatial visualization of top {0} recommended genes for spot {1}...".format(N, spot_id))

        df_rec = self.recommend_spot(spot_id, N)
        print(df_rec.head())

        vec = self.beta_embedded.loc[df_rec.index].values
        titles = df_rec['itemName']

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title, is_similar=True)
        else:
            pmf_plot.arrow(vec)

    def display_item(self, item, N=10, show_title=False, interactive=True):
        self._verify_load_status()
        if len(self.beta_embedded) == 0:
            self.tsne()
        title = item if isinstance(item, str) else self.item_title[item]
        print("Spatial visualization of top {0} similar genes for item {1}...".format(N, title))

        df_rec = self.recommend_items(item, N)
        print(df_rec.head())

        vec = self.beta_embedded.loc[df_rec.index].values
        titles = df_rec['itemName']

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title, is_similar=True)
        else:
            pmf_plot.arrow(vec)

    def display_genre(self, genre, N=10, show_title=False, interactive=True):
        self._verify_load_status()
        if len(self.beta_embedded) == 0:
            self.tsne()
        assert genre in self.genres, \
            "Genre {} doesen't exist in the dataset".format(genre)

        rand_ids = np.random.choice(list(self.genre_items[genre]), N)
        vec = self.beta_embedded.loc[rand_ids].values
        titles = pd.Series(rand_ids).map(self.item_title)

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title, is_similar=True)
        else:
            pmf_plot.arrow(vec)

    def display_joint(self, spot_id, iter=2, N=10, show_title=False, interactive=True):
        """Iteratively plot interacting spots & items"""
        self._verify_load_status()
        if len(self.beta_embedded) == 0:
            self.tsne()
        spot_ids, df_items = self.recommend_joint(spot_id, iter=iter)

        vec_spots = self.theta.loc[spot_ids].values
        vec_items = self.beta_embedded.loc[df_items.index].values
        titles = df_items['itemName']

        if interactive:
            pmf_plot.arrow_joint_interactive(vec_spots, vec_items, titles, show_title=show_title)
        else:
            pmf_plot.arrow_joint(vec_spots, vec_items)

    def display_random(self, N=3, n_neighbors=10, show_title=False, interactive=True):
        self._verify_load_status()
        if len(self.beta_embedded) == 0:
            self.tsne()
        print('Spatial visualization of the neighbors of {} random items'.format(N))

        rand_ids = np.random.choice(list(self.items), N)
        indices = set()
        for id in rand_ids:
            idx = self.recommend_items(id, n_neighbors, verbose=0).index
            indices = indices.union(idx)

        vec = self.beta_embedded.loc[indices].values
        titles = pd.Series(list(indices)).map(self.item_title)

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title)
        else:
            pmf_plot.arrow(vec)

"""
The :mod:`evaluate` module defines the :func:`evaluate` function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import time
import os
import math

import numpy as np
from six import iteritems
from six import itervalues

from surprise import accuracy, dump, Dataset, KNNBasic, Trainset, PredictionImpossible 
from surprise import similarities as sims
from surprise.prediction_algorithms.knns import SymmetricAlgo

class TemporalTrainset(Trainset):
    def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
                 offset, raw2inner_id_users, raw2inner_id_items,
                 newest_timestamp, oldest_timestamp):
        Trainset.__init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
                 offset, raw2inner_id_users, raw2inner_id_items)
        self.newest_timestamp = newest_timestamp
        self.oldest_timestamp = oldest_timestamp


    def all_ratings(self):
        """Generator function to iterate over all ratings.
        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """

        for u, u_ratings in iteritems(self.ur):
            for i, r, _ in u_ratings:
                yield u, i, r

def construct_temporal_trainset(raw_trainset):

    raw2inner_id_users = {}
    raw2inner_id_items = {}

    current_u_index = 0
    current_i_index = 0

    rm = defaultdict(int)
    ur = defaultdict(list)
    ir = defaultdict(list)

    newest_timestamp = 0
    oldest_timestamp = 1000000000000
    # user raw id, item raw id, rating, time stamp
    for urid, irid, r, timestamp in raw_trainset:
        try:
            uid = raw2inner_id_users[urid]
        except KeyError:
            uid = current_u_index
            raw2inner_id_users[urid] = current_u_index
            current_u_index += 1
        try:
            iid = raw2inner_id_items[irid]
        except KeyError:
            iid = current_i_index
            raw2inner_id_items[irid] = current_i_index
            current_i_index += 1
        timestamp = int(timestamp)
        newest_timestamp = max(newest_timestamp,timestamp)
        oldest_timestamp = min(oldest_timestamp,timestamp)
        rm[uid, iid] = r
        ur[uid].append((iid, r, timestamp))
        ir[iid].append((uid, r, timestamp))

    n_users = len(ur)  # number of users
    n_items = len(ir)  # number of items
    n_ratings = len(raw_trainset)

    trainset = TemporalTrainset(ur,
                        ir,
                        n_users,
                        n_items,
                        n_ratings,
                        (1,5),
                        0,
                        raw2inner_id_users,
                        raw2inner_id_items,
                        newest_timestamp,
                        oldest_timestamp)

    return trainset

class TemporalKNN(KNNBasic):
    def __init__(self, k=40, min_k=1, sim_options={'user_based':True,'name':'msd'}, epochs = 4, **kwargs):
        SymmetricAlgo.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k
        self.epochs = epochs

    def train(self, trainset):
        SymmetricAlgo.train(self, trainset)
        self.sim = self.compute_similarities()


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r, _) in self.yr[y]]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (_, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


    def compute_similarities(self):
        """Build the simlarity matrix.
    
        The way the similarity matric is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).
    
        Returns:
            The similarity matrix."""
    
        construction_func = {'cosine': sims.cosine,
                             'msd': sims.msd,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline}
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur
    
        min_support = self.sim_options.get('min_support', 1)
    
        global_newest = self.trainset.newest_timestamp
        global_oldest = self.trainset.oldest_timestamp
        global_delta = global_newest - global_oldest
        epoch_size = int(global_delta / self.epochs)
        print("global_newest=="+str(global_newest)+"  global_oldest=="+str(global_oldest)+" global_delta=="+str(global_delta))
        global_sim = np.zeros((n_x, n_x), np.double)
        epochs_tot = 0
        for epoch in xrange(1,self.epochs+1):
            epochs_tot += epoch
            oldest = global_oldest + (epoch_size * (epoch-1))
            newest = oldest + epoch_size
            print("newest=="+str(newest)+"  oldest=="+str(oldest)+" epochs_tot=="+str(epochs_tot))
            epoch_yr = {}
            epoch_yr_size = 0
            for y, y_ratings in iteritems(yr):
                epoch_yr[y] = [(r[0],r[1]) for r in y_ratings if oldest <= r[2] < (newest + 1)]
                epoch_yr_size += len(epoch_yr[y])
            print("epoch_yr_size=="+str(epoch_yr_size))
            args = [n_x, epoch_yr, min_support]
    
            name = self.sim_options.get('name', 'msd').lower()
            if name == 'pearson_baseline':
                shrinkage = self.sim_options.get('shrinkage', 100)
                #bu, bi = self.compute_baselines()
                if self.sim_options['user_based']:
                    bx, by = bu, bi
                else:
                    bx, by = bi, bu
        
                args += [self.trainset.global_mean, bx, by, shrinkage]
        
            try:
                print('Computing the {0} similarity matrix for epoch {1}...'.format(name,epoch))
                sim = construction_func[name](*args)
                print('Done computing similarity matrix.')
                np.multiply(sim, epoch)
                global_sim = np.add(global_sim,sim)
            except KeyError:
                raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                                'are ' + ', '.join(construction_func.keys()) + '.')
        return np.multiply(global_sim,1/float(epochs_tot))


def evaluate(algo, trainset, testset, measures=['rmse'], with_dump=False,
             dump_dir=None, verbose=1):
    """Evaluate the performance of the algorithm on given data.

    Depending on the nature of the ``data`` parameter, it may or may not
    perform cross validation.

    Args:
        algo(:obj:`AlgoBase <surprise.prediction_algorithms.bases.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on which
            to evaluate the algorithm.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module. Default is ``['rmse', 'mae']``.
        with_dump(bool): If True, the predictions, the trainsets and the
            algorithm parameters will be dumped for later further analysis at
            each fold (see :ref:`User Guide <dumping>`).  The file names will
            be set as: ``'<date>-<algorithm name>-<fold number>'``.  Default is
            ``False``.
        dump_dir(str): The directory where to dump to files. Default is
            ``'~/.surprise_data/dumps/'``.
        verbose(int): Level of verbosity. If 0, nothing is printed. If 1
            (default), accuracy measures for each folds are printed, with a
            final summary. If 2, every prediction is printed.

    Returns:
        A dictionary containing measures as keys and lists as values. Each list
        contains one entry per fold.
    """

    performances = CaseInsensitiveDefaultDict(list)
    print('Evaluating {0} of algorithm {1}.'.format(
          ', '.join((m.upper() for m in measures)),
          algo.__class__.__name__))
    print()

    if verbose:
        print('-' * 12)

    # train and test algorithm. Keep all rating predictions in a list
    algo.train(trainset)
    predictions = algo.test(testset, verbose=(verbose == 2))

    # compute needed performance statistics
    for measure in measures:
        f = getattr(accuracy, measure.lower())
        performances[measure].append(f(predictions, verbose=verbose))

    if with_dump:

        if dump_dir is None:
            dump_dir = os.path.expanduser('~') + '/.surprise_data/dumps/'

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
        file_name = date + '-' + algo.__class__.__name__
        file_name = os.path.join(dump_dir, file_name)

        dump(file_name, predictions, trainset, algo)

    if verbose:
        print('-' * 12)
        print('-' * 12)
        for measure in measures:
            print('Mean {0:4s}: {1:1.4f}'.format(
                  measure.upper(), np.mean(performances[measure])))
        print('-' * 12)
        print('-' * 12)

    return performances


class CaseInsensitiveDefaultDict(defaultdict):
    """From here:
        http://stackoverflow.com/questions/2082152/case-insensitive-dictionary.

        As pointed out in the comments, this only covers a few cases and we
        should override a lot of other methods, but oh well...

        Used for the returned dict, so that users can use perf['RMSE'] or
        perf['rmse'] indifferently.
    """
    def __setitem__(self, key, value):
        super(CaseInsensitiveDefaultDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDefaultDict, self).__getitem__(key.lower())

    def __str__(self):

        # retrieve number of folds. Kind of ugly...
        n_folds = [len(values) for values in itervalues(self)][0]

        row_format = '{:<8}' * (n_folds + 2)
        s = row_format.format(
            '',
            *['Fold {0}'.format(i + 1) for i in range(n_folds)] + ['Mean'])
        s += '\n'
        s += '\n'.join(row_format.format(
            key.upper(),
            *['{:1.4f}'.format(v) for v in vals] +
            ['{:1.4f}'.format(np.mean(vals))])
            for (key, vals) in iteritems(self))

        return s


def to_raw_fold(set_dict):
	raw_fold = []
	for key in set_dict:
		raw_fold.extend(set_dict[key])
	return raw_fold


data = Dataset.load_builtin('ml-1m')
ratings_by_user = {}
for r in data.raw_ratings:
	user = str(r[0])
	if user in ratings_by_user:
		ratings_by_user[user].append(r)
	else:
		ratings_by_user[user] = [r]

testset_by_user = {}
trainset_by_user = {}
for user in ratings_by_user:
	sorted_user_ratings = sorted(ratings_by_user[user], key=lambda x: x[3])
	length = len(sorted_user_ratings)
	first80perc = int(math.ceil(length*0.8))
	testset_by_user[user] = sorted_user_ratings[first80perc:]
	trainset_by_user[user] = sorted_user_ratings[:first80perc]


raw_trainset = to_raw_fold(trainset_by_user)
trainset = construct_temporal_trainset(raw_trainset)
raw_testset = to_raw_fold(testset_by_user)
testset = [(ruid, riid, r) for (ruid, riid, r, _) in raw_testset]

algo = TemporalKNN()

evaluate(algo, trainset, testset)

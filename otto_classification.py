#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from pandas import read_csv, DataFrame, Series, concat
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, svm, grid_search

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score

import pylab as pl
import matplotlib.pyplot as plt

import os

dirs = ['data_plot', 'test_plot']

def start():
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)

def plot_train():
    print "Plot data..."
    data = read_csv('./train.csv', sep = ',')
    for k in range(1, 94):
        param = 'feat_%s' % k
        data = data.sort([param])
        print "box plot %s..." % param.replace("_", " ")
        df = concat([data[param], data['target']], axis=1, keys=[param, 'target'])
        f = plt.figure(figsize=(8, 6))
        p = df.boxplot(by='target', ax = f.gca())
        img = './%s/feat_%s_class.png' % (dirs[0], k)
        f.savefig(img)

start()
plot_train()

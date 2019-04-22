import csv
import codecs
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
import nltk


# general function for reading in a csv file
def read_csv(filepath):
    with codecs.open(filepath, 'r', encoding='utf-8', errors='ignore') \
            as filestring:
        return [row for count, row in enumerate(csv.reader(filestring))
                if count != 0]


# meant to parse the tweets.csv file specifically
# extracts the sentiment score, confidence, and original text
def parse(data):
    parsed_data = []
    for row in data:
        parsed_row = []
        if row[1] == 'neutral':
            parsed_row.append(0)
        elif row[1] == 'positive':
            parsed_row.append(1)
        elif row[1] == 'negative':
            parsed_row.append(-1)
        else:
            continue

        parsed_row.append(float(row[2]))
        parsed_row.append(row[10])

        parsed_data.append(parsed_row)

    return parsed_data


# takes the parsed tweet data and separate it into polar or neutral tweets
# polar tweets are 1, neutral tweets are 0
def polar_neutral_split(data):
    polar_neutral_data = copy.deepcopy(data)

    for row in polar_neutral_data:
        row[0] = abs(row[0])

    return polar_neutral_data


# takes the parsed tweet data and removes neutral tweets
# neg tweets are now 0, pos tweets remain 1
def pos_neg_split(data):
    pos_neg_data = [copy.deepcopy(row) for row in data if row[0] != 0]

    for row in pos_neg_data:
        if row[0] == -1:
            row[0] = 0

    return pos_neg_data


# separates the text and the results
def generate_x_y(data):
    X = [row[2] for row in data]
    y = [row[0] for row in data]

    return X, y


# gets the accuracy of predicted y's against the value of true y's
def get_accuracy(true_y, pred_y):
    if len(pred_y) != len(true_y):
        raise('Invalid dimensions, true_y and pred_y do not match')

    total_correct = 0
    for count, output in enumerate(true_y):
        if output == pred_y[count]:
            total_correct += 1

    return total_correct / len(pred_y)


# create an instance of a classifier
def generate_classifier(classifier):
    # help on how to do this taken from:
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

    if classifier == 'sgd':
        clf = SGDClassifier(loss='hinge', penalty='l2')
    elif classifier == 'svm':
        clf = SVC(kernel='linear')
    elif classifier == 'nb':
        clf = MultinomialNB()
    else:
        raise('Specified unsupported classifer')

    tokenizer = nltk.tokenize.TweetTokenizer()
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenizer.tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', clf)
    ])


# create an instance of a regressor
def generate_regressor(regressor):
    if regressor == 'svm':
        reg = SVR(kernel='linear')
    else:
        raise('Specified unsupported regressor')

    tokenizer = nltk.tokenize.TweetTokenizer()
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenizer.tokenize)),
        ('tfidf', TfidfTransformer()),
        ('reg', reg)
    ])


# do a fold-fold cross validation using the specified classifier
def cross_validate(fold, is_classifier, predictor_type, X, y, thresholds=None):
    kf = KFold(fold)
    train_accs = []
    test_accs = []

    for train_index, test_index in kf.split(X):
        x_train = [X[index] for index in train_index]
        x_test = [X[index] for index in test_index]
        y_train = [y[index] for index in train_index]
        y_test = [y[index] for index in test_index]

        if is_classifier:
            predictor = generate_classifier(predictor_type)
        else:
            predictor = generate_regressor(predictor_type)

        predictor.fit(x_train, y_train)
        y_pred_train = predictor.predict(x_train)
        y_pred_test = predictor.predict(x_test)

        if is_classifier:
            final_y_pred_train = y_pred_train
            final_y_pred_test = y_pred_test
        else:
            final_y_pred_train = regression_to_classification(
                thresholds, y_pred_train)
            final_y_pred_test = regression_to_classification(
                thresholds, y_pred_test)

        train_accs.append(get_accuracy(y_train, final_y_pred_train))
        test_accs.append(get_accuracy(y_test, final_y_pred_test))

    return np.mean(train_accs), np.mean(test_accs)


# find which samples the svm struggled with
def find_errors(data, true_y, pred_y):
    if len(true_y) != len(pred_y):
        raise('Invalid dimensions, true_y and pred_y do not match')

    errors = []
    for i, output in enumerate(true_y):
        prediction = pred_y[i]
        if output != prediction:
            row = data[i]
            errors.append([i, row[2], output, prediction, row[1]])

    return errors


def get_error_metrics(true_y, pred_y):
    if len(true_y) != len(pred_y):
        raise('Invalid dimensions, true_y and pred_y do not match')

    # format: true value but prediction value
    # zero but positive, zero but negative
    # positive but negative, negative but positive
    # positive but zero, negative but zero
    zbp, zbn, pbn, pbz, nbp, nbz = 0, 0, 0, 0, 0, 0
    for i, output in enumerate(true_y):
        prediction = pred_y[i]
        if prediction == output:
            continue

        if output == 0:
            if prediction == 1:
                zbp += 1
            elif prediction == -1:
                zbn += 1
        elif output == 1:
            if prediction == -1:
                pbn += 1
            elif prediction == 0:
                pbz += 1
        elif output == -1:
            if prediction == 1:
                nbp += 1
            elif prediction == 0:
                nbz += 1

    return zbp, zbn, pbn, pbz, nbp, nbz


# out of all the wrong predictions, check the confidence of the classification
def check_confidence(data, true_y, pred_y):
    if len(true_y) != len(pred_y):
        raise('Invalid dimensions, true_y and pred_y do not match')

    confidences = []
    for i, output in enumerate(true_y):
        if output != pred_y[i]:
            confidences.append(data[i][1])

    return confidences


# turns the y's generated by a regression into a classification using the
# given thresholds
def regression_to_classification(thresholds, regressed_y):
    neg_threshold = thresholds[0]
    pos_threshold = thresholds[1]

    classified_y = []
    for y in regressed_y:
        if y < neg_threshold:
            classified_y.append(-1)
        elif y < pos_threshold:
            classified_y.append(0)
        else:
            classified_y.append(1)

    return classified_y


# make a histogram and save it
def make_histogram(title, x_label, y_label, arr):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.hist(arr, 'auto')
    plt.savefig('%s.jpg' % title)
    plt.show()
    plt.clf()

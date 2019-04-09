import csv
import codecs
import copy

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
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
def get_accuracy(true_y, y):
    if len(y) != len(true_y):
        raise('Invalid dimensions, true_y and y do not match')

    total_correct = 0
    for count, output in enumerate(true_y):
        if output == y[count]:
            total_correct += 1

    return total_correct / len(y)


# gets the mean of an array of numbers
def array_mean(arr):
    return sum(arr) / len(arr)


# create an instance of a stochastic gradient descent classifier
def generate_classifier(classifier):
    # help on how to do this taken from:
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

    if classifier == 'svm':
        clf = SGDClassifier(loss='hinge', penalty='l2',
                            alpha=1e-3, random_state=42,
                            max_iter=5, tol=None)
    elif classifier == 'naive_bayes':
        clf = MultinomialNB()
    else:
        raise('Specified unsupported classifer')

    return Pipeline([
        ('vect', CountVectorizer(tokenizer=nltk.word_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', clf)
    ])


# do a fold-fold cross validation using the specified classifier
def cross_validate(fold, classifier, X, y):
    kf = KFold(fold)
    train_accs = []
    test_accs = []

    for train_index, test_index in kf.split(X):
        x_train = [X[index] for index in train_index]
        x_test = [X[index] for index in test_index]
        y_train = [y[index] for index in train_index]
        y_test = [y[index] for index in test_index]

        classifier.fit(x_train, y_train)
        y_pred_train = classifier.predict(x_train)
        y_pred_test = classifier.predict(x_test)

        train_accs.append(get_accuracy(y_train, y_pred_train))
        test_accs.append(get_accuracy(y_test, y_pred_test))

    return array_mean(train_accs), array_mean(test_accs)


# adjust the scores to take confidence into account
# neutral scores might be hurt, bc they remain neutral with 100% confidence
# hard to account for bc don't know if crowd is stuck between pos/neg
# whereas can most likely assume unsure polar answers stuck bw polar/neutral
def adjust_scores(data):
    adjusted_scores = [[row[0] * row[1], row[2]] for row in data]

    return adjusted_scores

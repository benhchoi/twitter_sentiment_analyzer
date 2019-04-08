import csv
import codecs
import copy
import os


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


# adjust the scores to take confidence into account
# neutral scores might be hurt, bc they remain neutral with 100% confidence
# hard to account for bc don't know if crowd is stuck between pos/neg
# whereas can most likely assume unsure polar answers stuck bw polar/neutral
def adjust_scores(data):
    adjusted_scores = [[row[0] * row[1], row[2]] for row in data]

    return adjusted_scores

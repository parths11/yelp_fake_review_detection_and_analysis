
import pandas as pd
import sqlite3
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
from profanity_check import predict
import string


senti = sia()
tknzr = TweetTokenizer()
stopWords = set(stopwords.words('english'))

pos_family = {
    'noun' : ['NN', 'NNS', 'NNP', 'NNPS'],
    'pron' : ['PRP', 'PRP$', 'WP', 'WP$'],
    'verb' : ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'adj' :  ['JJ', 'JJR', 'JJS'],
    'adv' : ['RB', 'RBR', 'RBS', 'WRB']
}


def get_polarity(x):
    ss = TextBlob(str(x))
    return round(ss.sentiment.polarity, 4)


def get_subjectivity(x):
    ss = TextBlob(str(x))
    return round(ss.sentiment.subjectivity, 4)


def if_profane(x):
    try:
        y = predict([str(x)])
        return y[0]
    except ValueError:
        return 0


def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            pos = list(tup)[1]
            if pos in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt


def get_text_feat(df):

    df["polarity"] = df["reviewContent"].apply(get_polarity)
    df["subjectivity"] = df["reviewContent"].apply(get_subjectivity)

    df["profanity check"] = df["reviewContent"].apply(if_profane)

    df['tokens'] = df['reviewContent'].apply(lambda x: tknzr.tokenize(x))
    df['review_length'] = df['tokens'].apply(lambda x: len(x))
    df['no_of_stopwords'] = df['tokens'].apply(lambda x: len([i for i in x if i in stopWords]))
    df['char_count'] = df['reviewContent'].apply(len)
    df['word_density'] = df['char_count'] / (df['review_length'] + 1)

    df['punctuation_count'] = df['reviewContent'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    df['upper_case_word_count'] = df['reviewContent'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    df['noun_count'] = df['reviewContent'].apply(lambda x: check_pos_tag(x, 'noun'))
    df['verb_count'] = df['reviewContent'].apply(lambda x: check_pos_tag(x, 'verb'))
    df['adj_count'] = df['reviewContent'].apply(lambda x: check_pos_tag(x, 'adj'))
    df['adv_count'] = df['reviewContent'].apply(lambda x: check_pos_tag(x, 'adv'))
    df['pron_count'] = df['reviewContent'].apply(lambda x: check_pos_tag(x, 'pron'))

    df.drop("tokens", axis=1, inplace=True)
    df.drop("flagged", axis=1, inplace=True)


def main():
    con_res = sqlite3.connect("data/yelpResData.db")

    con_res.text_factory = bytes
    yelp_review_reviewContent = pd.read_sql("SELECT reviewID, reviewContent, flagged FROM review", con_res)
    pd.set_option('display.max_columns', None)

    yelp_review_reviewContent.reviewContent = yelp_review_reviewContent.reviewContent.str.decode('utf-8', errors='ignore')
    yelp_review_reviewContent.reviewID = yelp_review_reviewContent.reviewID.str.decode('utf-8', errors='ignore')
    yelp_review_reviewContent.flagged = yelp_review_reviewContent.flagged.str.decode('utf-8', errors='ignore')

    yelp_review_reviewContent = yelp_review_reviewContent[yelp_review_reviewContent['flagged'].isin(['N', 'Y'])]
    print(yelp_review_reviewContent.head())
    print("----------------------------------------------------")
    print("Shape of the Review data frame is: ", yelp_review_reviewContent.shape)

    get_text_feat(yelp_review_reviewContent)
    print(yelp_review_reviewContent.head())

    yelp_review_reviewContent.to_csv("reviews_text_features.csv")


if __name__ == "__main__":
    main()


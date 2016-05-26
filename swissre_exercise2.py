from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem.porter import *
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score

path = "/Downloads/bbc"
le = LabelEncoder()
stemmer = PorterStemmer()

def extract_content(topic, article_count):
    articles = []
    for i in range(1, article_count):
        article = ""
        try:
            with open(path + "/" + topic + "/" + str(i).zfill(3) + ".txt", "r") as f:
                for line in f.readlines():
                    article += line
            articles.append(article)
        except UnicodeDecodeError:
            continue
    return articles

def cleanup(line):
    words = []
    for l in line.split("\n"):
        for w in l.split(" "):
            if w not in stopwords.words('english') and re.match("[A-Za-z]+", w):
                words.append(stemmer.stem(w.lower()))
    return " ".join(words)


business_articles = extract_content("business", 511)
entertainment_articles = extract_content("entertainment", 387)
politics_articles = extract_content("politics", 418)
sport_articles = extract_content("sport", 512)
tech_articles = extract_content("tech", 402)

df = pd.DataFrame({
        "articles": business_articles + entertainment_articles + politics_articles + sport_articles + tech_articles,
        "topics": ["business"]*len(business_articles) + 
        ["entertainment"]*len(entertainment_articles) + 
        ["politics"]*len(politics_articles) + 
        ["sport"]*len(sport_articles) + 
        ["tech"]*len(tech_articles)
    })

vector = df['articles'].apply(cleanup)
count_vectorizer = CountVectorizer()
count_vectorizer.fit(vector)

X = count_vectorizer.transform(vector)
y = le.fit_transform(df['topics'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nb = MultinomialNB()
nb.fit(X_train, y_train)
probabs = nb.predict_proba(X_test)

print(f1_score(y_test, nb.predict(X_test)))
0.974604713072
print(accuracy_score(y_test, nb.predict(X_test)))
0.974550898204
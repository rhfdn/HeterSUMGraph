# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import argparse
import re
from nltk.tokenize import LineTokenizer, sent_tokenize, word_tokenize
import json

# %%
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# %%
# Parse args if script mode
parser = argparse.ArgumentParser(description='compute tfidf in sentence')

parser.add_argument('-input',type=str,default="./data/nyt50_sample.json")
parser.add_argument('-output',type=str,default="./data/nyt_sample_dataset_tfidf.json")
parser.add_argument('-docs_col_name',type=str,default="docs")
parser.add_argument('-is_sep_n',type=int,default=0,choices=[0,1])

args = None

if is_notebook():
    args = parser.parse_args("")
else:
    args = parser.parse_args()

# %%
def compute_tfidf_ws(s, vectorizer = TfidfVectorizer()):
  tfidf_values = vectorizer.fit_transform([s]).toarray()[0]
  words = vectorizer.get_feature_names_out()
  tfidf_dict = {word: tfidf_values[i] for i, word in enumerate(words)}
  return tfidf_dict

# %%
re_html = re.compile(r'<[^>]+>')

# %%
df = pd.read_json(args.input)

# %%
tfidf_dataset = []

# %%
doc = ""

for idx in df.index:
  doc = doc + df[args.docs_col_name][idx]

# remove html tags
doc = re_html.sub('', doc)

tfidfs = compute_tfidf_ws(doc)

# %%
with open(args.output, 'w') as f:
  json.dump(tfidfs, f)

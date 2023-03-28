# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import argparse
import re
from nltk.tokenize import LineTokenizer, sent_tokenize, word_tokenize

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

parser.add_argument('-input',type=str,default="../data/nyt50_sample.json")
parser.add_argument('-output',type=str,default="../data/nyt_sample_tfidf.json")
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
for idx in df.index:
  doc = df[args.docs_col_name][idx]
  # remove html tags
  doc = re_html.sub('', doc)
  # split doc
  if args.is_sep_n:
    nltk_line_tokenizer = LineTokenizer()
    doc = nltk_line_tokenizer.tokenize(doc)
  else:
    doc = sent_tokenize(doc)
  # for each sentence compute tfidf
  tfidf_doc = []
  for s in doc:
    try:
      tfidf_doc.append(compute_tfidf_ws(s=s))
    except:
      tfidf_doc.append(0)
  tfidf_dataset.append(tfidf_doc)

# %%
odf = pd.DataFrame({"tfidf": tfidf_dataset}, index=df.index)

# %%
odf.to_json(args.output)

# %%




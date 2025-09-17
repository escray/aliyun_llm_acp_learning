import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

import urllib.request

try:
  urllib.request.urlopen('https://raw.githubusercontent.com', timeout=5)
  print("network success")
except Exception as e:
  print("network failed: ", e)



# import nltk
# nltk.data.path.append('./nltk_cache')

# try:
#   tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#   print("punkt load success")
# except LookupError as e:
#   print("punkt load failed: ", e)

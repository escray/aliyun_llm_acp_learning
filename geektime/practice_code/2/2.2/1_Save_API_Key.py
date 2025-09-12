import os
from config.load_key import load_key
load_key()
print(f'''Your Config API Key is: {os.environ["DASHSCOPE_API_KEY"][:5]+"*"*5}''')

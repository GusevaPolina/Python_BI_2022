##############################################################################
# this homework explores the functionality of API and web-scraping in Python #
##############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

import requests
from bs4 import BeautifulSoup

# task 1: web-scraping of IMDB

# send a request and read an html content
response = requests.get("https://www.imdb.com/chart/top/?ref_=nv_mp_mv250")
soup = BeautifulSoup(response.content, "lxml")
df = pd.read_html(response.content, header=0)[0]
# let' rename the rating column
df = df.rename(columns={'IMDb Rating': 'rating'})
# let's filter columns by breaking one into rank, name, and year

df_filtered = df.iloc[:, [1,2]]

df_filtered['rank'] = df_filtered['Rank & Title'].apply(lambda x: int(x.split('. ')[0]))
name_year = df_filtered['Rank & Title'].apply(lambda x: re.split(r' (?=\(\d{4})', ' '.join(x.split(' ')[1:])))
df_filtered['name'] = [film[0].strip() for film in name_year]
df_filtered['year'] = [int(film[1][-5:-1]) for film in name_year]

df_filtered = df_filtered.iloc[:, 1:]

# find the mark responsible for viewers' rating
print(soup.find_all('strong')[0])

# and add a column with that
pattern = r'(\d[\d,]*) user ratings'
df_filtered['n_reviews'] = list(map(lambda x: int(re.search(pattern, str(x)).group(1).replace(',', '')), soup.find_all('strong')))

# Выведите топ-4 фильма по количеству оценок пользователей и количество этих оценок (1 балл)
df_filtered.sort_values('n_reviews', ascending=False).iloc[:4].loc[:, ['name', 'n_reviews']]

# Выведите топ-4 лучших года (по среднему рейтингу фильмов в этом году) и средний рейтинг (1 балл)
df_filtered.groupby('year').mean().sort_values('rating', ascending=False).loc[:, ['rating']].head(4)

# create a column with directors

html_of_dicks = soup.find_all(attrs={"class": "titleColumn"})
print(html_of_dicks[0])

pattern = r'title="(.*?)"'

dicks = map(lambda x: re.search(pattern, str(x)).group(1), html_of_dicks)
dicks = list(map(lambda x: x.split(', ')[0][:-7], dicks))

df_filtered['director'] = dicks

# Постройте отсортированный barplot, где показано количество фильмов из списка
# для каждого режисёра (только для режиссёров с более чем 2 фильмами в списке) (1 балл)

from collections import Counter

name_counts = Counter(dicks)
filtered_dict = {k: v for k, v in name_counts.items() if v > 2}
sorted_dict = dict(sorted(filtered_dict.items(), key=lambda x: x[1]))
print(sorted_dict)

plt.figure(figsize=(12,4))

sns.barplot(x=list(sorted_dict.keys()), y=list(sorted_dict.values()))
plt.title('The number of films per a popular director in top250', fontsize=14)

plt.xticks(rotation=90); plt.ylabel('number of films directed')
plt.yticks(range(min(sorted_dict.values()), 1 + max(sorted_dict.values())))
plt.ylim(bottom=min(sorted_dict.values())-1)
plt.grid(axis='y', which='both')

# Выведите топ-4 самых популярных режиссёра (по общему числу людей оценивших их фильмы) (2 балла)
df_filtered.groupby('director').sum().sort_values('n_reviews', ascending=False).index[:4]

# Сохраните данные по всем 250 фильмам в виде таблицы с колонками (name, rank, year, rating, n_reviews, director) в 
# любом формате (2 балла) 
df_filtered.reindex(columns=['name', 'rank', 'year', 'rating', 'n_reviews', 'director']).to_csv('top250_imbd.csv', index=False)
df_filtered.reindex(columns=['name', 'rank', 'year', 'rating', 'n_reviews', 'director'])


# task 2: a decorator as a telegram_logger that sends a message after a function finishes
from datetime import timedelta
import io
import requests
import sys
import time


def telegram_logger(chat_id):
    def decorator(func):
        def inner_function(*args, **kwargs):
            # steal stdout and stderr into variables
            original_out = sys.stdout
            original_err = sys.stderr

            # create an object to write stdout and stderr into
            output = io.StringIO()
            sys.stdout = output
            sys.stderr = output

            try:
                start = time.time()
                result = func(*args, **kwargs)
                delta_time = timedelta(seconds=time.time() - start)
                message = f"🎉 Function `{func.__name__}` successfully finished in `{delta_time}`"

            except Exception as e:
                message = f"😞 Function `{func.__name__}` failed with an exception:\n"
                message += f"\n`{type(e).__name__}: {str(e)}`"

            finally:
                # return stdout and stderr back to normal functioning
                sys.stdout = original_out
                sys.stderr = original_err

                # if there's a file to send -- send it
                if len(output.getvalue()):
                    url = f'https://api.telegram.org/bot{TOKEN}/sendDocument'
                    data = {'chat_id': chat_id, 'parse_mode': 'MarkdownV2', 'caption': message}

                    # Convert StringIO object to string and Encode the string as bytes
                    output_bytes = output.getvalue().encode('utf-8')
                    my_file_like_object = io.BytesIO(output_bytes)  # Create BytesIO object from bytes
                    my_file_like_object.name = f'{func.__name__}.log'

                    files = {'document': my_file_like_object}
                    response = requests.post(url, data=data, files=files)

                # otherwise, send just a message
                else:
                    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
                    data = {'chat_id': chat_id, 'parse_mode': 'MarkdownV2', 'text': message}
                    response = requests.post(url, data=data)

        return inner_function

    return decorator


CHAT_ID = ""  # <- INSERT URS
TOKEN = ""  # <- INSERT URS


@telegram_logger(CHAT_ID)
def good_function():
    print("This goes to stdout")
    print("And this goes to stderr", file=sys.stderr)
    time.sleep(2)
    print("Wake up, Neo")


@telegram_logger(CHAT_ID)
def bad_function():
    print("Some text to stdout")
    time.sleep(2)
    print("Some text to stderr", file=sys.stderr)
    raise RuntimeError("Ooops, exception here!")
    print("This text follows exception and should not appear in logs")


@telegram_logger(CHAT_ID)
def long_lasting_function():
    time.sleep(2)


good_function()

bad_function()

long_lasting_function()

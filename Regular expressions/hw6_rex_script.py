#################################################################################################################
################### this homework explores the functionality of regular expressions in Python ###################
#################################################################################################################

import re
import matplotlib.pyplot as plt

# 1. Parse the references file using regular expressions and write all ftp links into the ftps file (5 points)

# download a file with ftp links
# !wget https://raw.githubusercontent.com/Serfentum/bf_course/master/15.re/references

# read the whole file with links
with open('references') as ref:
    ref_read = ref.read()

# create a pattern to retrieve all links
pattern_ref = r'ftp\.[./#\w]*'

# write the result into a output file
with open('ftps', 'w') as ftps:
    print(*re.findall(pattern_ref, ref_read), file=ftps, sep='\n')


# 2. Extract all numbers (consisting of only digits) from the story 2430 A.D. (5 points)

# download a file with the story
# !wget https://raw.githubusercontent.com/Serfentum/bf_course/master/15.re/2430AD

# read the whole story
with open('2430AD') as tale:
    tale_read = tale.read()

# create a pattern to retrieve all numbers
pattern_tale = r'([0-9]+)(\.[0-9]+)?'

# write the result into a output file
with open('task2', 'w') as numbers:
    print(*[i[0] for i in re.findall(pattern_tale, tale_read)], file=numbers)


# 3. From the same story, extract all the words containing the 'a' letter ignoring the case (5 points)

# create a pattern to retrieve all words with the 'a' letter
pattern_tale_Aa = r'\w*[aA]\w*'

# write the result into a output file
with open('task3', 'w') as Aa:
    print(*re.findall(pattern_tale_Aa, tale_read), file=Aa)


# 4. Extract all exclamatory sentences from the story (5 points)

# create a pattern to retrieve all exclamatory sentences
pattern_tale_exclamation = r'[a-zA-Z \,]*!'

# write the result into a output file
with open('task4', 'w') as exclamation:
    print(*re.findall(pattern_tale_exclamation,
                      tale_read), file=exclamation)


# 5. Plot a histogram of the length distribution of unique words from the story (regarding a case) (5 points)

# create a pattern to retrieve all words
pattern_words = r"[a-zA-Z0-9-']+"

# find all words
all_words = re.findall(pattern_words, tale_read)
# unify them
all_words_lower = list(map(lambda x: x.lower(), all_words))
# create a set of unique words
all_words_unique = sorted(set(all_words_lower), key=len)

# for each of the word length count its frequency
length_frequency = {}
for i in range(len(all_words_unique[0]), len(all_words_unique[-1]) + 1):
    length_frequency[i] = len(list(filter(lambda x: len(x) == i,
                                          all_words_unique))) / len(all_words_unique)

# plot the result
plt.figure(figsize=(8, 6))
plt.bar(length_frequency.keys(), length_frequency.values(), width=1)
plt.xticks(range(len(all_words_unique[0]), len(all_words_unique[-1])))
plt.title('Length distribution of unique words\nfrom Buy Jupiter and Other Stories by Isaac Asimov')
plt.xlabel('a length of a word')
plt.ylabel('frequency')


# 6. Add after each Russian vowel 'K + an uppercase vowel' (5 points)

# input your string to transform
desirable_string = input('Insert ur string: ')

# a list of Russian lower- and uppercase vowels
list_vowels = 'УЕАОЭЁЯИЮЫ'
list_vowels = list_vowels + list_vowels.lower()

# iterate over all vowels starting with the upper once
# since the added part contains a tail with a capital vowel
for i in list_vowels:
    desirable_string = re.sub(f'{i}', f'{i}К{i.upper()}', desirable_string)


# 7. Make a function to extract sentences with a given number of words from the text
# (considering prepositions and conjunctions as words) (5 extra points)

def find_n_words_sentences(one_string, number):
    # recognise %user_number% separate words
    pattern_dop = '(([а-яА-Я]+ ?){' + str(number) + '})'
    matched = re.findall(pattern_dop, one_string)
    # create a desirable output format
    for i in range(len(matched)):
        matched[i] = tuple(matched[i][0].split())
    return matched

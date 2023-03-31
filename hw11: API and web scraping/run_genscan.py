from bs4 import BeautifulSoup
import pandas as pd
import re
import requests



def exon_table_maker(soup):
  soup_split = str(soup).split('\n\n')
  soup_header = [string for string in soup_split if string.startswith('Gn.Ex')][0]

  exon_table = pd.DataFrame()
  for string in soup_split[soup_split.index(soup_header)+3:]:
    if not string:
      break
    GnEx, Type, S = string.split('  ')[0].split()
    Begin, End = string.split('  ')[1], string.split('  ')[2]
    exon_table = pd.concat([exon_table, pd.DataFrame([GnEx, Type, S, Begin, End]).T])

  exon_table = exon_table.rename(columns=dict(zip(range(5), soup_header.split()[:5])))

  return exon_table


def intron_table_maker(exon_table):
  intron_table = pd.DataFrame([list(exon_table['Gn.Ex'])[:-1],
                               list(exon_table['...End'])[:-1],
                               list(exon_table['.Begin'])[1:]],
                               index=['NN', 'Begin', 'End']).T

  return intron_table


class GenscanOutput():
  def __init__(self, response):
    self.status = response.status_code

    soup = BeautifulSoup(response.content)
    # I know it's only for one possible sequence not a list
    self.cds_list = str(soup).split('_aa')[1].split('</pre')[0].replace("\n", "")

    exon_table = exon_table_maker(soup)
    self.intron_list = exon_table
    self.exon_list = intron_table_maker(exon_table)


def run_genscan(sequence=None, sequence_file=None, organism="Vertebrate", 
                exon_cutoff=1.00, sequence_name=""):
  data = {'-s': sequence, '-u': sequence_file, '-o': organism,
          '-e': exon_cutoff, '-n': sequence_name,
          '-p': 'Predicted peptides only'} # the obligatory thing
  url = 'http://hollywood.mit.edu/cgi-bin/genscanw_py.cgi'
  response = requests.post(url, data=data)
  return GenscanOutput(response)

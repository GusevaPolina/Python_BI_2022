#####################################################################################
# this homework explores the functionality of object-oriented programming in Python #
#####################################################################################

# task 1: connected classes Chat, Message and User

import datetime


class Chat:
    def __init__(self):
        self.chat_history = []

    def show_last_message(self):
        # the assumption is made that we live in the one timeline
        # therefore, the last message is one sent as the latest
        last_message = self.chat_history[0]
        return last_message.show()

    def get_history_from_time_period(self, start, stop):
        chat_sorted = Chat()  # type(self) ?

        for message in self.chat_history[::-1]:
            if start <= message.datetime <= stop:
                chat_sorted.recieve(message)

        return chat_sorted

    def show_chat(self):
        for message in self.chat_history:
            message.show()
            print()

    def recieve(self, message):
        self.chat_history = [message] + self.chat_history


class Message:
    def __init__(self, text, date_time, user, ability='is a god/ess'):
        self.text = text
        self.datetime = datetime.datetime.strptime(date_time, "%d/%m/%y %H:%M")
        self.user = User(ability=ability, name=user)

    def show(self):
        print(self.datetime, self.user.name, self.text)

    def send(self, chat):
        chat.recieve(self)


class User:
    def __init__(self, name, ability):
        self.ability = ability
        self.name = name

    def greeting(self):
        print(f'Hi, {self.name}! U r gorgeous')

    def description(self):
        print(f'{self.name} {self.ability}')


# task 2: new syntax for Python

# to avoid annoying nested list let's flatten them
def flatten_list(nested_list):
    flattened_list = []
    for element in nested_list:
        if isinstance(element, list):
            flattened_list += flatten_list(element)
        else:
            flattened_list.append(element)
    return flattened_list


class Args:
    def __init__(self, *args, **kwargs):
        self.args = list(args)
        self.args = flatten_list(self.args)
        self.kwargs = list(kwargs.values())

        self.all_argies = self.args + self.kwargs

    def __rlshift__(self, other):
        return other(*self.all_argies)


# task 3: a new way to pass arguments to functions

class StrangeFloat(float):
    def __getattr__(self, attr):
        if '__' + attr.split('_')[0][:3] + '__' in dir(float):
            fnc = getattr(float, '__' + attr.split('_')[0][:3] + '__')
            return fnc(self, float(attr.split('_')[1]))
        else:
            dict_attr = {'subtract': self - float(attr.split('_')[1]),
                         'multiply': self * float(attr.split('_')[1]),
                         'divide': self / float(attr.split('_')[1])}
            return StrangeFloat(dict_attr[attr.split('_')[0]])

        def __repr__(self):
            return float(self)


# task 4: rewrite changing to dunders

# overall, 3 (three) dunders
import numpy as np

matrix2 = []
for idx in range(0, 100, 10):
    matrix2.__iadd__([list(range(idx, idx + 10))])

selected_columns_indices = list(filter(lambda x: x in range(1, 5, 2), range(matrix.__len__())))
selected_columns = map(lambda x: [x.__getitem__(col) for col in selected_columns_indices], matrix)

arr = np.array(list(selected_columns))

mask = arr[:, 1] % 3 == 0
new_arr = arr[mask]

product = new_arr @ new_arr.T

if (product[0] < 1000).all() and (product[2] > 1000).any():
    print(product.mean())


# task 5:

from abc import ABC, abstractmethod


class BiologicalSequence(ABC):
    def __init__(self, sequence):
        self.sequence = sequence.upper()

    def __len__(self):
        return len(self.sequence)

    @abstractmethod
    def __getitem__(self, slc):
        pass

    @abstractmethod
    def check_alphabet(self):
        pass

    def __repr__(self):
        return str(self.sequence)


class NucleicAcidSequence(BiologicalSequence):
    def check_alphabet(self):
        check_na = (['U'], ['T'])[int(self.sequence_type == "DNA")]
        check_set = set(['A', 'G', 'C'] + check_na)
        len_foreign = len(set(self.sequence) - check_set)
        assert len_foreign == 0, 'wrong alphabet!!!'

    def complement(self):
        if self.sequence_type == "DNA":
            dict_trans = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        elif self.sequence_type == 'RNA':
            dict_trans = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
        return ''.join([dict_trans[nucleotid] for nucleotid in self.sequence])

    def gc_content(self):
        gc_count = self.sequence.count('G') + self.sequence.count('C')
        return gc_count / len(self.sequence)

    def __getitem__(self, slc):
        if isinstance(slc, int):
            return self.sequence.__getitem__(slc)
        elif isinstance(slc, slice):
            return self.sequence.__getitem__(slc)


class DNASequence(NucleicAcidSequence):
    def __init__(self, param):
        super().__init__(param)
        self.sequence_type = "DNA"

    def transcribe(self):
        dict_trans_dna = {'A': 'U', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join([dict_trans_dna[nucleotid] for nucleotid in self.sequence])


class RNASequence(NucleicAcidSequence):
    def __init__(self, param):
        super().__init__(param)
        self.sequence_type = "RNA"


class AminoAcidSequence(BiologicalSequence):
    def pI_calculator(self):
        dict_pI = {'A': 6.11, 'R': 10.76, 'N': 5.43, 'D': 2.98, 'C': 5.15,
                   'E': 3.08, 'Q': 5.65, 'G': 6.06, 'H': 7.64, 'I': 6.04,
                   'L': 6.04, 'K': 9.47, 'M': 5.71, 'F': 5.76, 'P': 6.30,
                   'S': 5.70, 'T': 5.60, 'W': 5.88, 'Y': 5.63, 'V': 6.02}

        all_pI = [dict_pI[aa] for aa in self.sequence]
        return sum(all_pI) / len(all_pI)

    def __getitem__(self, slc):
        if isinstance(slc, int):
            return self.sequence.__getitem__(slc)
        elif isinstance(slc, slice):
            return self.sequence.__getitem__(slc)

    def check_alphabet(self):
        check_aa = {'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'}
        len_foreign = len(set(self.sequence) - check_aa)
        assert len_foreign == 0, 'wrong alphabet!!!'

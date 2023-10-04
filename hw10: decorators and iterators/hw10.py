##################################################################################
# this homework explores the functionality of decorators and iterators in Python #
##################################################################################


# task 1: a new iteration for a dictionary

class MyDict(dict):

    def __iter__(self):
        return ((key, value) for key, value in self.items())


# task 2: add to an iterator

def iter_append(iterator, item):
    yield from iterator
    yield item


# task 3: a class that returns itself

# it can be put inside classes as well
# as class could be inserted automatically and universally
def decorator_class_dominus(class_name):
    def decorator(func):
        def inner_func(*args, **kwargs):
            func_output = func(*args, **kwargs)
            return class_name(func_output)

        return inner_func

    return decorator


# normally we would use @classmethod
class MyString(str):
    @decorator_class_dominus(class_name=MyString)
    def reverse(self):
        return self[::-1]

    @decorator_class_dominus(class_name=MyString)
    def make_uppercase(self):
        return "".join([chr(ord(char) - 32) if 97 <= ord(char) <= 122 else char for char in self])

    @decorator_class_dominus(class_name=MyString)
    def make_lowercase(self):
        return "".join([chr(ord(char) + 32) if 65 <= ord(char) <= 90 else char for char in self])

    @decorator_class_dominus(class_name=MyString)
    def capitalize_words(self):
        return " ".join([word.capitalize() for word in self.split()])


class MySet(set):
    def is_empty(self):
        return len(self) == 0

    def has_duplicates(self):
        return len(self) != len(set(self))

    @decorator_class_dominus(class_name=MySet)
    def union_with(self, other):
        return self.union(other)

    @decorator_class_dominus(class_name=MySet)
    def intersection_with(self, other):
        return self.intersection(other)

    @decorator_class_dominus(class_name=MySet)
    def difference_with(self, other):
        return self.difference(other)


# task 4: a decorator that switches privacy

my_variable = None


def switch_privacy(cls):
    global my_variable
    if not my_variable:
        cls = cls()

    def inner_function():

        global my_variable
        if not my_variable:
            my_variable = True

            list_public, list_private = [], []

            # first, define all public and __private attributes
            for attr in cls.__dir__():
                # if public
                if callable(getattr(cls, attr)) and not attr.startswith("_"):
                    list_public.append(attr)
                # if private (but NOT dunder)
                if callable(getattr(cls, attr)) and attr.startswith("_ExampleClass__"):
                    list_private.append(attr)

            # second, rename them all and delete previous versions
            for attr_private in list_private:  # if private -> turn public
                # setattr(obj, attr_name, value)
                # value = getattr(obj, attr_name)
                setattr(cls, attr_private[len('_ExampleClass__'):],
                        getattr(cls, attr_private))
                # delattr(obj, attr_name)
                delattr(cls.__class__, attr_private)  # [len('_ExampleClass'):])

            for attr_public in list_public:  # if public -> private
                setattr(cls, "_ExampleClass__" + attr_public, getattr(cls, attr_public))
                delattr(cls.__class__, attr_public)

        return cls

    return inner_function


@switch_privacy
class ExampleClass:
    def public_method(self):
        return 1

    def _protected_method(self):
        return 2

    def __private_method(self):
        return 3

    def __dunder_method__(self):
        pass


# task 5: a new context manager that returns a FastaRecord class

from dataclasses import dataclass


@dataclass
class FastaRecord:
    seq: str
    id_: str
    description: str


class OpenFasta:

    def __init__(self, path):
        self.path = path
        self.header = None
        self.position = None
        self.end_of_file = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read_record(self):
        with open(self.path) as fasta_file:
            # let's go to the position where we stop reading last time
            if self.position:
                fasta_file.seek(self.position)
            # let's check if that's the first ever header
            if not self.header:
                self.header = fasta_file.readline().strip()

            # read the first line of a sequence
            first_line = fasta_file.readline().strip()
            # add it to the line storage
            line_storage = [first_line]
            # read the next line
            line = fasta_file.readline().strip()
            # keep read and store line til a next header
            while not line.startswith('>'):
                line_storage.append(line)
                line = fasta_file.readline().strip()
                if not line:
                    self.end_of_file = True
                    break

            # create a FastaRecord object
            seq = ''.join(line_storage)
            id_, *description = self.header.split()
            one_read = FastaRecord(seq=seq, id_=id_,
                                   description=' '.join(description))

            # to not forget the header, write it into guts
            self.header = line
            self.position = fasta_file.tell()

            return one_read

    def read_records(self):
        # let's get to the beginning of the file
        self.position, self.end_of_file = 0, False

        record_storage = []
        while not self.end_of_file:
            record_storage.append(self.read_record())

        # let's get to the beginning of the file again
        self.position, self.end_of_file = 0, False

        return record_storage


# task 6: genotype combinations

# subtask 1

ploidy = 2
p1, p2 = 'Aabb', 'Aabb'

p1_allele = [p1[i:i + ploidy] for i in range(0, len(p1), ploidy)]
p2_allele = [p2[i:i + ploidy] for i in range(0, len(p2), ploidy)]

import itertools

combinations_list = []

for i in range(len(p1_allele)):
    combinations = list(itertools.product(p1_allele[i], p2_allele[i]))
    combinations = [sorted(t) for t in combinations]
    combinations_list.append(combinations)

all_combinations = list(itertools.product(*combinations_list))
all_combinations = [''.join([''.join(sublst) for sublst in lst]) for lst in all_combinations]
print(*all_combinations, sep='\n')


# subtask 2

def get_offspting_genotype_probability(parent1, parent2, target_genotype, ploidy=2):
    p1_allele = [parent1[i:i + ploidy] for i in range(0, len(parent1), ploidy)]
    p2_allele = [parent2[i:i + ploidy] for i in range(0, len(parent2), ploidy)]

    combinations_list = []

    for i in range(len(p1_allele)):
        combinations = list(itertools.product(p1_allele[i], p2_allele[i]))
        combinations = [sorted(t) for t in combinations]
        combinations_list.append(combinations)

    all_combinations = list(itertools.product(*combinations_list))
    all_combinations = [''.join([''.join(sublst) for sublst in lst]) for lst in all_combinations]
    counter = all_combinations.count(target_genotype)

    return counter / len(all_combinations)


print(get_offspting_genotype_probability(parent1="Aabb", parent2="Aabb", target_genotype="Aabb"))

# subtask 3

# as we need ONLY UNIQUE
# than we can just replace all variaty of alleles by the needed ones

p0 = 'АаБбВвГгДдЕеЖжЗзИиЙйКкЛл'
p0_dict = {}
for i in range(0, len(p0), 2):
    p0_dict[p0[i].upper()] = p0[i:i + 2]

p1 = 'АаБбввГгДдЕеЖжЗзИиЙйккЛлМмНн'
p2 = 'АаббВвГгДДЕеЖжЗзИиЙйКкЛлМмНН'

p1_allele = [p1[i:i + ploidy] for i in range(0, len(p1), ploidy)]
p2_allele = [p1[i:i + ploidy] for i in range(0, len(p2), ploidy)]

combinations_list = []

for i in range(len(p1_allele)):
    if p1_allele[i][0].upper() in p0.upper():
        combinations = [list(p0_dict[p1_allele[i][0].upper()])]
    else:
        combinations = list(itertools.product(p1_allele[i], p2_allele[i]))
        combinations = [sorted(t) for t in combinations]
    combinations_list.append(combinations)

all_combinations = list(itertools.product(*combinations_list))
all_combinations = [''.join([''.join(sublst) for sublst in lst]) for lst in all_combinations]

print(set(all_combinations))


# subtask 4

# the same as subtask 2



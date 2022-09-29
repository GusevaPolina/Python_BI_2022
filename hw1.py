# #################################################################################################################
# the script will allow u to transcribe, reverse, complement, or reverse & complement ur favourite RNA or DNA seq #
# #################################################################################################################

# the command options
def options():
    print('''\n   Please, choose wisely from the following options.

  List of valid commands: 
    exit                is for quiting the program (do it now or never), 
    transcribe          is for transcription of the sequence, 
    reverse             is for reverse of the sequence,
    complement          is for complement of the sequence, 
    reverse complement  is for reverse complement of the sequence,
    F                   is for pay respect.

    ''')
    return input('Please enter your command, my lord/my lady/my noble non-binary person: ')


# the command functions
complementator_dna = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G',
                      'a': 't', 'g': 'c', 't': 'a', 'c': 'g'}
complementator_rna = {'A': 'U', 'G': 'C', 'U': 'A', 'C': 'G',
                      'a': 'u', 'g': 'c', 'u': 'a', 'c': 'g'}

transcriptor_dna_to_rna = {'A': 'U', 'G': 'C', 'T': 'A', 'C': 'G',
                           'a': 'u', 'g': 'c', 't': 'a', 'c': 'g'}
transcriptor_rna_to_dna = {'A': 'T', 'G': 'C', 'U': 'A', 'C': 'G',
                           'a': 't', 'g': 'c', 'u': 'a', 'c': 'g'}

allowed_set = set(list(complementator_dna.keys()) + ['u', 'U'])


def transcribe(sequence):
    if 'u' in sequence.lower():
        processed_seq = [transcriptor_rna_to_dna[i] for i in sequence]
    else:
        processed_seq = [transcriptor_dna_to_rna[i] for i in sequence]
    return ''.join(processed_seq)


def reverse(sequence):
    return sequence[::-1]


def complement(sequence):
    if 'u' in sequence.lower():
        processed_seq = [complementator_rna[i] for i in sequence]
    else:
        processed_seq = [complementator_dna[i] for i in sequence]
    return ''.join(processed_seq)


def reverse_complement(sequence):
    processed_seq = complement(sequence)[::-1]
    return ''.join(processed_seq)


commandor = {'transcribe': transcribe, 'reverse': reverse,
             'complement': complement, 'reverse complement': reverse_complement}

# the program core
print('Hi stranger, u r welcome to the sequence translation script!\n')
answer = options().lower().strip()

while answer != 'exit':
    if answer == 'f':
        print('Lots of respect, dude')
    elif answer not in commandor:
        print('Ups, choose a valid command if u want to work')
    else:
        seq = input('Please enter your sequence: ')
        if not set(seq).issubset(allowed_set):
            print('Ups, u have unexpected character/s in ur sequence:',
                  *set(seq).difference(allowed_set))
        elif 'u' in seq.lower() and 't' in seq.lower():
            print('Ups, invalid sequence: uracil and thymine in one place!')
        else:
            print(f'Ur processed sequence is:  {commandor[answer](seq)}\n')

    answer = options().lower().strip()

print('\nBye bye')

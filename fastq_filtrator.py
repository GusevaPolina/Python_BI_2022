######################################################################################################################
# the second homework: just functions to filter (gc, length, phred) fastq files and rewrite into separate output/s   #
######################################################################################################################

# return matching of GC content into some boundaries
def gc_filter(example, boundaries=(0, 100)):
    g_number = example.lower().count('g')
    c_number = example.lower().count('c')
    gc_score = 100 * (g_number + c_number) / len(example)

    if type(boundaries) is float or type(boundaries) is int:
        return gc_score <= boundaries
    elif len(boundaries) == 2:
        return boundaries[0] <= gc_score <= boundaries[1]


# return matching of sequence length into some boundaries
def length_filter(example, boundaries=(0, 2 ** 32)):
    if type(boundaries) is float or type(boundaries) is int:
        return len(example) <= boundaries
    elif len(boundaries) == 2:
        return boundaries[0] <= len(example) <= boundaries[1]


# return matching of sequence quality by the phred33 scale into some boundaries
def quality_filter(example, threshold=0):
    sum_q_score = sum(map(lambda x: ord(x) - 33, example))
    quality = sum_q_score / len(example)

    if type(threshold) is float or type(threshold) is int:
        return quality >= threshold
    elif len(threshold) == 2:
        return threshold[0] <= quality <= threshold[1]


# the main function check every sequence in an input file
# and sort it into a passed by all filters file or (optionally) to a file with FAILURES (aka Asian parents file)
def main(input_fastq, output_file_prefix, save_filtered=False,
         gc_bounds=(0, 100), length_bounds=(0, 2 ** 32), quality_threshold=0):

    with open(input_fastq) as f:
        read_lines = f.readlines()

    for i in range(0, len(read_lines), 4):
        q_answer = quality_filter(read_lines[i + 3].strip(), threshold=quality_threshold)
        l_answer = length_filter(read_lines[i + 1].strip(), boundaries=length_bounds)
        gc_answer = gc_filter(read_lines[i + 1].strip(), boundaries=gc_bounds)

        if q_answer and l_answer and gc_answer:
            with open(output_file_prefix + "_passed.fastq", 'a') as output_right:
                for j in read_lines[i:i + 4]:
                    output_right.write(j)

        else:
            if save_filtered:
                with open(output_file_prefix + "_failed.fastq", 'a') as output_looser:
                    for j in read_lines[i:i + 4]:
                        output_looser.write(j)
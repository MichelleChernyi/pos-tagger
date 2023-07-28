import os
import sys
import argparse
import numpy as np

TAGS = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']

def read_from_file(files):
    words = []
    words_set = {}
    word_tags = []
    tags_num = [0 for _ in range(91)]
    tags_num2 = [0 for _ in range(91)]
    words_set = {}
    sentences_num = 0
    words_num = 0
    
    last_tag = None
    for file in files:
        tags_file = open(file, "r")
        for line in tags_file:
            words_num +=1
            line = line.rsplit(" : ")
            word = line[0].strip()
            # all the words in training model in order they appear
            words.append(word.lower())
        
            tag = line[1].strip()
            # all the tags in training model in order they appear
            word_tags.append(tag)

            # num of each tag in whole training set
            if word not in ['.', '?', '!']:
                tags_num2[TAGS.index(tag)] += 1
            last_tag = tag
            last_word = word

            tags_num[TAGS.index(tag)] += 1
            

            if word in ['.', '?', '!']:
                sentences_num +=1

        if last_word not in ['.', '?', '!']:
            tags_num2[TAGS.index(last_tag)] -= 1

    words_set = list(dict.fromkeys(words))

    return words, words_set, word_tags, tags_num, tags_num2, sentences_num, words_num

def create_prob_tables(words, words_set, word_tags, tags_num, tags_num2, sentences_num, words_num):
    M = np.array([[0.00001 for _ in range(91)] for _ in range(len(words_set))], dtype='f')
    T = np.array([[0.00001 for _ in range(91)] for _ in range(91)], dtype='f')

    count1 = 0
    prev = None
    
    I = np.array([0.00001 for _ in range(91)], dtype='f')

    for i, word in enumerate(words):
        if i+1 < words_num and word not in ['.', '!', '?']:
            if T[TAGS.index(word_tags[i])][TAGS.index(word_tags[i + 1])]  == 0.00001:
                T[TAGS.index(word_tags[i])][TAGS.index(word_tags[i + 1])] =  10 / tags_num2[TAGS.index(word_tags[i])]
            else:
                T[TAGS.index(word_tags[i])][TAGS.index(word_tags[i + 1])] +=  10 / tags_num2[TAGS.index(word_tags[i])]

        if  M[words_set.index(word)][TAGS.index(word_tags[i])] == 0.00001 and tags_num[TAGS.index(word_tags[i])] != 0:
            M[words_set.index(word)][TAGS.index(word_tags[i])] = 10 / tags_num[TAGS.index(word_tags[i])]
        elif tags_num[TAGS.index(word_tags[i])] != 0:
            M[words_set.index(word)][TAGS.index(word_tags[i])] += 10 / tags_num[TAGS.index(word_tags[i])]

        if (prev in ['.', '?', '!']) or not prev:
            if I[TAGS.index(word_tags[i])] == 0.00001:
                I[TAGS.index(word_tags[i])] = 10 /sentences_num
            else:
                I[TAGS.index(word_tags[i])] += 10 / sentences_num
            count1+=1

        prev = word
    return I, M, T

def viterbi(E, I, T, M, word_set): 
    #column / inside each list - S - set of tags 
    #row / list nums - E - set of observations, first word, second word, etc.

    prob = np.array([[0.00001 for _ in range(91)] for _ in range(len(E))], dtype='f')
    prev = np.array([[0 for _ in range(91)] for _ in range(len(E))], dtype='f')

    for i in range(91):
        if E[0].lower() in word_set:
            index = word_set.index(E[0].lower())
            prob[0][i] = I[i] * M[index][i]
        else:
            prob[0][i] = I[i] 
        prev[0][i] = None
    
    for t in range(1, len(E)):
        for i in range(91):
            if E[t].lower() in word_set:
                index = word_set.index(E[t].lower())
                x = prob[t-1] * T[:, i] * M[index][i]
                x = np.argmax(x)
                prob[t][i] = prob[t-1][x] * T[x][i] *M[index][i]
                prev[t][i] = x
            else:
                x = prob[t-1] * T[:, i] 
                x = np.argmax(x)
                prob[t][i] = prob[t-1][x] * T[x][i] 
                prev[t][i] = x

    return prob, prev

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )

    parser.add_argument(
        "--answerfile",
        type=str,
        required=False,
        help="The answer file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))
    if args.answerfile: 
        print("answer file is {}".format(args.answerfile))


    print("Starting the tagging process.")
    E = []
    test_file = open(args.testfile, "r")
    new_lst = []
    check_next = False
    for line in test_file:
        if line.strip() in ['.', '!', '?']:
            new_lst.append(line.strip())
            check_next= True
            E.append(new_lst)
            new_lst = []
        else:
            new_lst.append(line.strip())


    words, words_set, word_tags, tags_num, tags_num2, sentences_num, words_num= read_from_file(training_list)
    I, M, T= create_prob_tables(words,words_set, word_tags, tags_num, tags_num2, sentences_num, words_num)
    all_all_tags = []
    for lst in E:
        prob, prev = viterbi(lst, I, T, M, words_set)
        max = 0
        ind = 0

        for j, proby in enumerate(prob[len(prob)-1]):
            if proby > max:
                max = proby
                ind = j
            
        prev_ind = ind
        all_tags = []
        for i in range(len(prev) - 1, 0, -1):
            word = lst[i]
            previous_tag = TAGS[prev_ind]
            full = word + ' : ' + previous_tag
            all_tags.insert(0, full)
            prev_ind = int(prev[i][prev_ind])
        previous_tag = TAGS[prev_ind]
        word = lst[0]
        full = word + ' : ' + previous_tag
        all_tags.insert(0, full)
        all_all_tags.extend(all_tags)
        
    output_file = open(args.outputfile, "w")
    for line in all_all_tags:
        output_file.write(line)
        output_file.write("\n")

    answerfile = args.testfile.split("_")[0] + "_train.txt"
    if args.answerfile: 
        a, b = open(args.outputfile).readlines(), open(args.answerfile).readlines()
        print(f"Accuracy: {str(1 - (len([i for i in range(len(a)) if a[i] != b[i]]) / len(a)))[0:7]}%")

from pprint import pprint

f = open('data/bison/bison_captions_val')
lines = f.readlines()
n_sentences = len(lines)

longest = 0
tot_words = 0
shortest = 999
counters = dict()
for line in lines:
# for i in range(128):
    # line = lines[i]
    l = line[:-1]
    l = l.split()
    
    tot_words += len(l)
    if len(l) > longest:
        longest = len(l)
    if len(l) == 40:
        print(l)
    if len(l) < shortest:
        shortest = len(l)

    if not len(l) in counters:
        counters[len(l)] = 0
    counters[len(l)] += 1

print("Number of real sentences : {0}".format(n_sentences))
print("Longest sentence : {0} words".format(longest))
print("Shortest sentence : {0} words".format(shortest))
print("Total words in real sentences : {0}".format(tot_words))
print("Total words in generated sentences : {0} * {1} = {2}".format(n_sentences, longest, n_sentences*longest))
pprint(counters)

'''
{7: 26,
 8: 545,
 9: 847,
 10: 981, 
 11: 882,
 12: 690,
 13: 496,
 14: 312,
 15: 206,
 16: 125,
 17: 89,
 18: 54,
 19: 41,
 20: 41,
 21: 19,
 22: 26,
 23: 10,
 24: 4,
 25: 8,
 26: 2,
 27: 4,
 28: 3,
 29: 3,
 30: 3,
 31: 1,
 32: 2,
 33: 1,
 36: 2,
 38: 1,
 40: 1}
 '''
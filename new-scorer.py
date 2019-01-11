import json
import sys


def accuracy(gold_list, auto_list):
    gold_sense_list = [relation['Sense'][0] for relation in gold_list]
    auto_sense_list = [relation['Sense'][0] for relation in auto_list]

    correct = len([1 for i in range(len(gold_list))
        if gold_sense_list[i] == auto_sense_list[i]])

    print('Accuracy: {:<13.5}'.format(correct / len(gold_list)), end='\n\n')

def prf_for_one_tag(gold_list, auto_list, tag):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(gold_list)):
        if gold_list[i]['Sense'][0] == tag and auto_list[i]['Sense'][0] == tag:
            tp += 1
        elif gold_list[i]['Sense'][0] == tag:
            fn += 1
        elif auto_list[i]['Sense'][0] == tag:
            fp += 1

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r) if p + r != 0 else '-'
    print('{:35} precision {:<13.5}recall {:<13.5}F1 {:<13.5}'.format(tag, p, r, f))

    return tp, fp, fn, p, r, f

def prf(gold_list, auto_list):
    tag_dict = {relation['Sense'][0]:None for relation in gold_list}

    total_tp, total_fp, total_fn, total_p, total_r, total_f = 0, 0, 0, 0, 0, 0
    for tag in tag_dict: 
        tp, fp, fn, p, r, f = prf_for_one_tag(gold_list, auto_list, tag)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_p += p
        total_r += r
        total_f += f if f != '-' else 0

    print()
    p = total_tp / (total_tp + total_fp)
    r = total_tp / (total_tp + total_fn)
    f = 2 * p * r / (p + r) if p + r != 0 else '-'
    print('{:35} precision {:<13.5}recall {:<13.5}F1 {:<13.5}'.format('Micro-Average', p, r, f))
    #print('{:35} precision {:<13.5}recall {:<13.5}F1 {:<13.5}'.format(
    #    'Macro-Average', total_p / len(tag_dict), r / len(tag_dict), f / len(tag_dict)))
    print()

if __name__ == '__main__':
    gold = sys.argv[1]
    auto = sys.argv[2]

    gold_list = [json.loads(x) for x in open(gold)]
    auto_list = [json.loads(x) for x in open(auto)]

    print('='*60 + '\nEvaluation for all discourse relations\n')
    accuracy(gold_list, auto_list)
    prf(gold_list, auto_list)

    print('='*60 + '\nEvaluation for explicit discourse relations only\n')
    accuracy([g for g in gold_list if g['Type'] == 'Explicit'],
        [a for a in auto_list if a['Type'] == 'Explicit'])
    prf([g for g in gold_list if g['Type'] == 'Explicit'],
        [a for a in auto_list if a['Type'] == 'Explicit'])

    print('='*60 + '\nEvaluation for non-explicit discourse relations only\n')
    accuracy([g for g in gold_list if g['Type'] != 'Explicit'],
        [a for a in auto_list if a['Type'] != 'Explicit'])
    prf([g for g in gold_list if g['Type'] != 'Explicit'],
        [a for a in auto_list if a['Type'] != 'Explicit'])

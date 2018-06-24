#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('csv_path', type=str)
parser.add_argument('gbdt_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

def gen_hashed_fm_feats(feats, nr_bins):
	# format  field:hash_value:1
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

frequent_feats = read_freqent_feats(args['threshold'])

with open(args['out_path'], 'w') as f:
    for row, line_gbdt in zip(csv.DictReader(open(args['csv_path'])), open(args['gbdt_path'])):
        feats = []
		# gen_feats  数值特征取对数离散化 格式:I13-4 C16-234234234
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
			# 低频类别特征特殊编码
            if type == 'C' and feat not in frequent_feats:
                feat = feat.split('-')[0]+'less'
            if type == 'C':
                field += 13
            feats.append((field, feat))
		# gbdt 特征
        for i, feat in enumerate(line_gbdt.strip().split()[1:], start=1):
            field = i + 39
            feats.append((field, str(i)+":"+feat))

        feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')

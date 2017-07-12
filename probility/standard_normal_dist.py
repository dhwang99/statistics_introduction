# encoding: utf8

import pdb
import json 
import sys
import bisect

normal_dist_json_file_name = 'standard_normal_dist.json'

normal_dist_table = None
normal_dist_table_x = None
normal_dist_table_p = None

def load_from_json(jfile=normal_dist_json_file_name):
    global normal_dist_table_p, normal_dist_table_x, normal_dist_table

    normal_dist_table = json.load(open(jfile), 'r')
    normal_dist_table_x = map(lambda xp:xp[0], normal_dist_table)
    normal_dist_table_p = map(lambda xp:xp[1], normal_dist_table)

    return None

'''
给定p的情况下，计算x
'''
def guess_x(p):
    index = bisect.bisect_left(normal_dist_table_p, p)
    if index >= len(normal_dist_table_p):
        index -= 1
    return normal_dist_table_x[index]
    

'''
给定x的情况下，计算p
'''
def guess_p(x):
    index = bisect.bisect_left(normal_dist_table_x, x)
    if index >= len(normal_dist_table_x):
        index -= 1
    return normal_dist_table_p[index]

'''
从标准高斯表中倒入数据, x in R+, p in R+
'''
def convert_data_to_json(table_file_name, json_file_name):
    part_table = []
    with open(table_file_name, 'r') as f:
        line =f.readline()
    
        while True:
            line = f.readline()
            if line == '':
                break
    
            substep = 0.00
            items = line.split(' ')
    
            base = float(items[0])
            id = 0
            for s in items[1:]:
                x = base + 0.01 * id
                #pdb.set_trace()
                p = float(s) + 0.5
                id += 1
    
                part_table.append((x, p)) 
   
    negative_table = list(part_table[1:])
    negative_table.reverse()
    for id in range(len(negative_table)):  #remove zero point
        x_p = negative_table[id]
        negative_table[id] = (-1 * x_p[0], 1 - x_p[1]) 
    
    normal_dist_table = negative_table + part_table

    json.dump(normal_dist_table, open(json_file_name, 'w'))
    load_from_json(json_file_name)

    return None
   
#pdb.set_trace()
load_from_json()

if __name__ == "__main__":
    convert_data_to_json(sys.argv[1], normal_dist_json_file_name)

    p_s = 0.5
    x_s = 0.0
    x = guess_x(p_s)
    p = guess_p(x_s)

    print "real x,p vs normal_dist x,p: (%s, %s) vs (%s, %s)" % (x_s, p_s, x, p)

    p_s = 0.0013499
    x_s = -3.0
    x = guess_x(p_s)
    p = guess_p(x_s)

    print "real x,p vs normal_dist x,p: (%s, %s) vs (%s, %s)" % (x_s, p_s, x, p)

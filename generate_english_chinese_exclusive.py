import pandas as pd
import numpy as np
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description = 'description')
    parser.add_argument('--raw_file', help='description')
    parser.add_argument('--out_e',default='0', help='description')
    parser.add_argument('--out_c',default='0', help='description')
    args = parser.parse_args()
    
    with open(args.raw_file, 'r', encoding='utf8') as f:
        vocabularies = f.readlines()
    vocabularies = [i.strip() for i in vocabularies]
    
    en_letter = '[\u0041-\u005a|\u0061-\u007a|\s]+' # 大小写英文字母
    zh_char = '[\u4e00-\u9fa5]+' # 中文字符
    spl = '[\u0041-\u005a|\u0061-\u007a][\s]+[\u4e00-\u9fa5]' # 英文任意空格中文
    list_ENG = []
    list_CHN = []
    for i in vocabularies:
        x = re.search(spl,i)
        if x:
            nod_left = x.span()[0]+1
            nod_right = x.span()[1]-1
            list_ENG.append(i[:nod_left])
            list_CHN.append(i[nod_right:])
    
    with open(args.out_e,'w',encoding='utf-8') as f:
        for i in list_ENG:
            f.write(i + '\n')
            f.write('\n')
        
    with open(args.out_c,'w',encoding='utf-8') as f:
        for i in list_CHN:
            f.write(i + '\n')
            f.write('\n')
            

if __name__ == '__main__':
    main()
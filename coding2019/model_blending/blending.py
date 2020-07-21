#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os

def blending(out_file, in_file1, in_wg1, in_file2, in_wg2):
    print("%s, %s, %f, %s, %f\n" % (out_file, in_file1, in_wg1, in_file2, in_wg2))    

    with open(out_file, "w") as fo:
        with open(in_file1,'r') as f1:
            with open(in_file2,'r') as f2:
                for x, y in zip(f1.readlines(),f2.readlines()):
                    fo.write("%6f\n" % (float(x)*in_wg1 + float(y)*in_wg2))

if __name__ == "__main__":
    
    if len(sys.argv) < 5 or len(sys.argv)%2 ==0 :
        print("Useage: python blending.py file1, weight1, file2, weight2, ...\n")
        sys.exit(-1)

    in_file1 = sys.argv[1]
    in_wg1 = float(sys.argv[2])
    in_file2 = sys.argv[3]
    in_wg2 = float(sys.argv[4])
    out_file = "./blend_result_1.txt"
    blending(out_file, in_file1, in_wg1, in_file2, in_wg2)

    for i in range(len(sys.argv)):
        if i > 4 and i%2 == 1 :
            in_file1 = sys.argv[i]
            in_wg1 = float(sys.argv[i+1])
            in_file2 = out_file
            in_wg2 = 1
            out_file = "blend_result_%d.txt" % i
            blending(out_file, in_file1, in_wg1, in_file2, in_wg2)
    
    os.system("mv %s blend_result.txt" % out_file)
    os.system("rm -rf blend_result_*")

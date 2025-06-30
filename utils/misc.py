import os
import re 

exceptions = ['AA-RM', 'BC-JH', 'ED-GG', 'GR-JW', 'KT-MA', 'SL-LPR']
f54 = ["BC-JH","GR-JW"]

def correct(p):
    correct_dict = {
        "LP":"LPR",
        "PR":"LPR",
        "HK":"HKG",
        "KG":"HKG",
        "cD":"ED",
        "cC":"EC"
    }
    return correct_dict.get(p, p)

# This code returns whether a motion sequence corresponds to the CareGiver or CareReceiver
# ( File name formatting was not consistent across old dataset and new dataset )

def get_info(sub_f, f):
    concerned = ""

    dd = {}
        
    if "(P1)" in sub_f and "(P2)" in sub_f:
        o_split_with   = "P1"
        o_split_with_2 = "P2"
        split_with = "(" + o_split_with + ")"
        split_with_2 = "(" + o_split_with_2 + ")"
        splits = sub_f.split(split_with)
        splits2 = sub_f.split(split_with_2)
        dd[o_split_with] = splits[0].strip()[-2:]
        dd[o_split_with_2] = splits2[0].strip()[-2:]
    elif "(CG)" in sub_f:
        o_split_with   = "CG"
        o_split_with_2 = "CR"
        split_with = "(" + o_split_with + ")"
        split_with_2 = "(" + o_split_with_2 + ")"
        splits = sub_f.split(split_with)
        splits2 = sub_f.split(split_with_2)
        dd[o_split_with] = splits[0].strip()[-2:]
        dd[o_split_with_2] = splits2[0].strip()[-2:]
    else:
        o_split_with   = "CG"
        o_split_with_2 = "CR"
        split_with = "_" + o_split_with
        split_with_2 = "_" + o_split_with_2
        splits = sub_f.split(split_with)
        splits2 = sub_f.split(split_with_2)
        dd[o_split_with] = splits2[0].strip()[1:3]
        dd[o_split_with_2] = splits2[0].strip()[1:3]

    # if f in f54:
    last_split_with = "_5"
    # else:
    # last_split_with = "_50"

    try:
        concerned = correct(sub_f.split(last_split_with)[-2].strip().replace("(","").replace(")","")[-2:])
    except:
        breakpoint()
    inv = {}
    for k in dd.keys():
        dd[k] = correct(dd[k])
        inv[dd[k]] = k

    try:
        tmp = inv[correct(concerned)]
    except:
        breakpoint()
    return tmp


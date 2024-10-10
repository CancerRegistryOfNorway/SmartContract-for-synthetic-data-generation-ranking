#!/usr/bin/env python3

# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

# developed by Mohammad H. Tabatabaei and Narasimha Raghavan  

'''
Command line interface for synthrank TF.
Parses command line arguments and passes to the synthrankClient class
to process.
'''

import argparse
import logging
import os
import sys
import traceback
import datetime
import re
import pandas as pd

from colorlog import ColoredFormatter
from synthrank_client import synthrankClient

KEY_NAME = 'synthrank'

# hard-coded for simplicity (otherwise get the URL from the args in main):
DEFAULT_URL = 'http://localhost:8008'
# For Docker:
# DEFAULT_URL = 'http://rest-api:8008'

def create_console_handler(verbose_level):
    '''Setup console logging.'''
    del verbose_level # unused
    clog = logging.StreamHandler()
    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s %(levelname)-8s%(module)s]%(reset)s "
        "%(white)s%(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        })

    clog.setFormatter(formatter)
    clog.setLevel(logging.DEBUG)
    return clog

def setup_loggers(verbose_level):
    '''Setup logging.'''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(create_console_handler(verbose_level))

def create_parser(prog_name):
    '''Create the command line argument parser for the smartmed CLI.'''
    parent_parser = argparse.ArgumentParser(prog=prog_name, add_help=False)

    parser = argparse.ArgumentParser(
        description='Provides subcommands to manage your queries',
        parents=[parent_parser])

    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    subparsers.required = True

    regmethod_subparser = subparsers.add_parser('regmethod',
                                           help='populate the ledger with a synthetic data generation method',
                                           parents=[parent_parser])                                           
    regmethod_subparser.add_argument('method',
                                type=str,
                                help='type one of these: im, bn, mpom, clgp, mc_medgan, mice_lr, mice_lr_desc, mice_dt')
    regmethod_subparser.add_argument('--pcd',
                               type=float,
                               help='type a float value like 0.9')
    regmethod_subparser.add_argument('--lc',
                               type=float,
                               help='type a float value like 0.9')                            
    regmethod_subparser.add_argument('--crrs',
                               type=float,
                               help='type a float value like 0.9')
    regmethod_subparser.add_argument('--crsr',
                               type=float,
                               help='type a float value like 0.9')
    regmethod_subparser.add_argument('--sc',
                               type=float,
                               help='type a float value like 0.9')
    regmethod_subparser.add_argument('--ad',
                               type=float,
                               help='type a float value like 0.9')
    regmethod_subparser.add_argument('--mdp',
                               type=float,
                               help='type a float value like 0.9')
    regmethod_subparser.add_argument('--mdr',
                               type=float,
                               help='type a float value like 0.9')

    subparsers.add_parser('methods',
                                           help='display all the methods',
                                           parents=[parent_parser])

    delmethod_subparser = subparsers.add_parser('delmethod',
                                          help='delete a registered method',
                                          parents=[parent_parser])
    delmethod_subparser.add_argument('method',
                               type=str,
                               help='Method that is going to be deleted')

    compute_subparser = subparsers.add_parser('compute',
                                          help='Compute the total QoS score of the registered methods for a given purpose',
                                          parents=[parent_parser])
    compute_subparser.add_argument('purpose',
                               type=str,
                               help='Purpose that is used for the methods to compute the score based on it')
    compute_subparser.add_argument('m1',
                               type=str,
                               help='1st method that you want to compute its QoS based on the given scenario')
    compute_subparser.add_argument('m2',
                               type=str,
                               help='2nd method that you want to compute its QoS based on the given scenario')
    compute_subparser.add_argument('m3',
                               type=str,
                               help='3rd method that you want to compute its QoS based on the given scenario')
    compute_subparser.add_argument('m4',
                               type=str,
                               help='4th method that you want to compute its QoS based on the given scenario')
    compute_subparser.add_argument('m5',
                               type=str,
                               help='5th method that you want to compute its QoS based on the given scenario')
    compute_subparser.add_argument('m6',
                               type=str,
                               help='6th method that you want to compute its QoS based on the given scenario')
    compute_subparser.add_argument('m7',
                               type=str,
                               help='7th method that you want to compute its QoS based on the given scenario')
    compute_subparser.add_argument('m8',
                               type=str,
                               help='8th method that you want to compute its QoS based on the given scenario')

    rank_subparser = subparsers.add_parser('rank',
                                           help='display method ranks in a given scenario',
                                           parents=[parent_parser])
    rank_subparser.add_argument('purpose',
                               type=str,
                               help='purpose that its methods going to be ranked') 

    subparsers.add_parser('ranks',
                                           help='display all method ranks',
                                           parents=[parent_parser])                                                                                                                                    

    method_subparser = subparsers.add_parser('method',help='Swithching to the file execution mode for registering methods', parents=[parent_parser])
    method_subparser.add_argument('filepath',
                               type=str,
                               help='Path to the input.txt file')
    method_subparser.add_argument('--key',
                                type=str,
                                help='the ID of the user that is going to provide the file')
    
    qos_subparser = subparsers.add_parser('qos',help='Swithching to the file execution mode for computing qos', parents=[parent_parser])
    qos_subparser.add_argument('filepath',
                               type=str,
                               help='Path to the compute.txt file')

    qi_subparser = subparsers.add_parser('qi',help='Getting the file for registering QIs', parents=[parent_parser])
    qi_subparser.add_argument('filepath',
                               type=str,
                               help='Path to the qi.txt file')
    qi_subparser.add_argument('--key',
                                type=str,
                                help='the ID of the user that is going to provide the file')

    data_utility_subparser = subparsers.add_parser('data_utility',
                                           help='populate the ledger with data utility names',
                                           parents=[parent_parser])                                           
    data_utility_subparser.add_argument('qis',
                               nargs='*',
                                    type=str,
                                    help='type all data utility QIs separated by spaces')

    data_privacy_subparser = subparsers.add_parser('data_privacy',
                                           help='populate the ledger with data privacy names',
                                           parents=[parent_parser])                                           
    data_privacy_subparser.add_argument('qis',
                               nargs='*',
                                    type=str,
                                    help='type all data privacy QIs separated by spaces')

    subparsers.add_parser('qis',
                                           help='display all the QIs',
                                           parents=[parent_parser])

    cw_subparser = subparsers.add_parser('cw',help='Getting the file for registering category weights', parents=[parent_parser])
    cw_subparser.add_argument('filepath',
                               type=str,
                               help='Path to the weights.txt file')
    cw_subparser.add_argument('--key',
                                type=str,
                                help='the ID of the user that is going to provide the file') 

    category_weight_subparser = subparsers.add_parser('category_weight',
                                           help='populate the ledger with a purpose and its weights',
                                           parents=[parent_parser])                                           
    category_weight_subparser.add_argument('purpose',
                                type=str,
                                help='purpose name like: PurposeA')
    category_weight_subparser.add_argument('--data_utility',
                               type=float,
                               help='type a float value like 0.5, 1.5, 1')
    category_weight_subparser.add_argument('--data_privacy',
                               type=float,
                               help='type a float value like 0.5, 1.5, 1')

    subparsers.add_parser('cws',
                                           help='display all the category weights',
                                           parents=[parent_parser])

    wmp_subparser = subparsers.add_parser('wmp',help='Getting the file for registering wm plus weights', parents=[parent_parser])
    wmp_subparser.add_argument('filepath',
                               type=str,
                               help='Path to the WM_plus.txt file')
    wmp_subparser.add_argument('--key',
                                type=str,
                                help='the ID of the user that is going to provide the file')

    WM_plus_subparser = subparsers.add_parser('WM_plus',
                                         help='populate the ledger with a purpose and its weights',
                                         parents=[parent_parser])
    WM_plus_subparser.add_argument('purpose',
                                     type=str,
                                     help='purpose name like: PurposeA')
    WM_plus_subparser.add_argument('--PCD',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_plus_subparser.add_argument('--CRRS',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')                                                   
    WM_plus_subparser.add_argument('--CRSR',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_plus_subparser.add_argument('--LC',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_plus_subparser.add_argument('--SC',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_plus_subparser.add_argument('--AD',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_plus_subparser.add_argument('--MDP',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_plus_subparser.add_argument('--MDR',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')

    subparsers.add_parser('wmps',
                                           help='display all the WM plus weights',
                                           parents=[parent_parser])

    wmm_subparser = subparsers.add_parser('wmm',help='Getting the file for registering wm minus weights', parents=[parent_parser])
    wmm_subparser.add_argument('filepath',
                               type=str,
                               help='Path to the WM_minus.txt file')
    wmm_subparser.add_argument('--key',
                                type=str,
                                help='the ID of the user that is going to provide the file')

    WM_minus_subparser = subparsers.add_parser('WM_minus',
                                         help='populate the ledger with a purpose and its weights',
                                         parents=[parent_parser])
    WM_minus_subparser.add_argument('purpose',
                                     type=str,
                                     help='purpose name like: PurposeA')
    WM_minus_subparser.add_argument('--PCD',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_minus_subparser.add_argument('--CRRS',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')                                                   
    WM_minus_subparser.add_argument('--CRSR',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_minus_subparser.add_argument('--LC',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_minus_subparser.add_argument('--SC',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_minus_subparser.add_argument('--AD',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_minus_subparser.add_argument('--MDP',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')
    WM_minus_subparser.add_argument('--MDR',
                                     nargs='?',
                                     type=float,
                                     help='type a float value like 0.5, 1.5, 1')

    subparsers.add_parser('wmms',
                                           help='display all the WM plus weights',
                                           parents=[parent_parser])

    audit_subparser = subparsers.add_parser('audit',help='Auditing', parents=[parent_parser])
    audit_subparser.add_argument('--key',
                                type=str,
                                help='the ID of the user that is going to audit')

    subparsers.add_parser('isConsistent',
                                           help='display if there is consistency',
                                           parents=[parent_parser])                                                                                             

    return parser

def _get_private_keyfile(key_name):
    '''Get the private key for key_name.'''
    home = os.path.expanduser("~")
    key_dir = os.path.join(home, ".sawtooth", "keys")
    return '{}/{}.priv'.format(key_dir, key_name)

def do_data_utility(args, client):
    '''Subcommand to populate the ledger with data utility QIs. Calls client class to do the registering.'''
    start_time = datetime.datetime.now()
    response = client.reg_data_utility(args.qis)
    end_time = datetime.datetime.now()
    print("Find Response: {}".format(response))
    delay(start_time, end_time)

def do_data_privacy(args, client):
    '''Subcommand to populate the ledger with data privacy QIs. Calls client class to do the registering.'''
    start_time = datetime.datetime.now()
    response = client.reg_data_privacy(args.qis)
    end_time = datetime.datetime.now()
    print("Find Response: {}".format(response))
    delay(start_time, end_time)

def do_qis():
    '''Subcommand to show the QIs. Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = []
    for txs in client.qis():
        txs_decoded = txs.decode().split('|')
        for tx in txs_decoded:
            query_list.append(tx.split(','))
    end_time = datetime.datetime.now()
    if query_list is not None: 
        print("Data utility: "+ str(query_list[1]).replace("\\","").replace("\"","").replace("'","").replace("[","").replace("]","").replace(" ",""))
        print("Data privacy: "+ str(query_list[0]).replace("\\","").replace("\"","").replace("'","").replace("[","").replace("]","").replace(" ",""))    
    else:
        raise Exception("Transaction data not found")
    delay(start_time, end_time)
    return query_list

def do_category_weight(args, client):
    '''Subcommand to populate the ledger with category weights. Calls client class to do the registering.'''
    start_time = datetime.datetime.now()
    response = client.reg_category_weight(args.purpose, args.data_utility, args.data_privacy)
    end_time = datetime.datetime.now()
    print("Find Response: {}".format(response))
    delay(start_time, end_time)

def do_cws():
    '''Subcommand to show the list of category weights.  Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = [
        tx.split(',')
        for txs in client.cws()
        for tx in txs.decode().split('|')
    ]
    end_time = datetime.datetime.now()
    if query_list is not None:
        sorted_query_list = sorted(query_list, key=lambda x: x[0])
        for tx_data in sorted_query_list:
            purpose, utility, privacy = tx_data
            purpose = purpose.replace("'","").replace("[","").replace("]","")
            utility = utility.replace("'","").replace("[","").replace("]","")
            privacy = privacy.replace("'","").replace("[","").replace("]","")
            print(purpose + ") data utility: " + str(utility) +" | data privacy: " + str(privacy))
    else:
        raise Exception("Transaction data not found")
    delay(start_time, end_time)
    return sorted_query_list

def do_wm_plus(args, client):
    '''Subcommand to populate the ledger with wm plus weights. Calls client class to do the registering.'''
    start_time = datetime.datetime.now()
    response = client.reg_wm_plus(args.purpose, args.PCD, args.LC, args.CRRS, args.CRSR, args.SC, args.AD, args.MDP, args.MDR)
    end_time = datetime.datetime.now()
    print("Find Response: {}".format(response))
    delay(start_time, end_time)

def do_wmps():
    '''Subcommand to show the list of wm plus weights.  Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = [
        tx.split(',')
        for txs in client.wmps()
        for tx in txs.decode().split('|')
    ]
    end_time = datetime.datetime.now()
    if query_list is not None:
        sorted_query_list = sorted(query_list, key=lambda x: x[0])
        for tx_data in sorted_query_list:
            purpose, PCD, LC, CRRS, CRSR, SC, AD, MDP, MDR = tx_data
            purpose = purpose.replace("'","").replace("[","").replace("]","")
            PCD = PCD.replace("'","").replace("[","").replace("]","")
            LC = LC.replace("'","").replace("[","").replace("]","")
            CRRS = CRRS.replace("'","").replace("[","").replace("]","")
            CRSR = CRSR.replace("'","").replace("[","").replace("]","")
            SC = SC.replace("'","").replace("[","").replace("]","")
            AD = AD.replace("'","").replace("[","").replace("]","")
            MDP = MDP.replace("'","").replace("[","").replace("]","")
            MDR = MDR.replace("'","").replace("[","").replace("]","")
            print(purpose + ") PCD: " + str(PCD) +" | LC: " + str(LC) +" | CRRS: " + str(CRRS) +" | CRSR: " + str(CRSR) +" | SC: " + str(SC) +" | AD: " + str(AD) +" | MDP: " + str(MDP) +" | MDR: " + str(MDR))
    else:
        raise Exception("Transaction data not found")
    delay(start_time, end_time)
    return sorted_query_list

def do_wm_minus(args, client):
    '''Subcommand to populate the ledger with wm minus weights. Calls client class to do the registering.'''
    start_time = datetime.datetime.now()
    response = client.reg_wm_minus(args.purpose, args.PCD, args.LC, args.CRRS, args.CRSR, args.SC, args.AD, args.MDP, args.MDR)
    end_time = datetime.datetime.now()
    print("Find Response: {}".format(response))
    delay(start_time, end_time)

def do_wmms():
    '''Subcommand to show the list of wm minus weights.  Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = [
        tx.split(',')
        for txs in client.wmms()
        for tx in txs.decode().split('|')
    ]
    end_time = datetime.datetime.now()
    if query_list is not None:
        sorted_query_list = sorted(query_list, key=lambda x: x[0])
        for tx_data in sorted_query_list:
            purpose, PCD, LC, CRRS, CRSR, SC, AD, MDP, MDR = tx_data
            purpose = purpose.replace("'","").replace("[","").replace("]","")
            PCD = PCD.replace("'","").replace("[","").replace("]","")
            LC = LC.replace("'","").replace("[","").replace("]","")
            CRRS = CRRS.replace("'","").replace("[","").replace("]","")
            CRSR = CRSR.replace("'","").replace("[","").replace("]","")
            SC = SC.replace("'","").replace("[","").replace("]","")
            AD = AD.replace("'","").replace("[","").replace("]","")
            MDP = MDP.replace("'","").replace("[","").replace("]","")
            MDR = MDR.replace("'","").replace("[","").replace("]","")
            print(purpose + ") PCD: " + str(PCD) +" | LC: " + str(LC) +" | CRRS: " + str(CRRS) +" | CRSR: " + str(CRSR) +" | SC: " + str(SC) +" | AD: " + str(AD) +" | MDP: " + str(MDP) +" | MDR: " + str(MDR))
    else:
        raise Exception("Transaction data not found")
    delay(start_time, end_time)
    return sorted_query_list

def do_regmethod(args, client):
    '''Subcommand to populate the ledger with a synthetic data generation method. Calls client class to do the registering.'''
    start_time = datetime.datetime.now()
    response = client.regmethod(args.method, args.pcd, args.lc, args.crrs, args.crsr, args.sc, args.ad, args.mdp, args.mdr)
    end_time = datetime.datetime.now()
    print("Find Response: {}".format(response))
    delay(start_time, end_time)

def do_delmethod(args):
    '''Subcommand to delete a method.  Calls client class to do the deleting.'''
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    response = client.delmethod(args.method)
    print("delete Response: {}".format(response))

def do_methods():
    '''Subcommand to show the list of methods.  Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = [
        tx.split(',')
        for txs in client.methods()
        for tx in txs.decode().split('|')
    ]
    end_time = datetime.datetime.now()
    if query_list is not None:
        sorted_query_list = sorted(query_list, key=lambda x: x[0].split("'")[1])
        count = 0;
        for tx_data in sorted_query_list:
            count = count + 1
            method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr = tx_data
            method = method.replace("'","").replace("[","").replace("]","")
            pcd = pcd.replace("'","").replace("[","").replace("]","")
            lc = lc.replace("'","").replace("[","").replace("]","")
            crrs = crrs.replace("'","").replace("[","").replace("]","")
            crsr = crsr.replace("'","").replace("[","").replace("]","")
            sc = sc.replace("'","").replace("[","").replace("]","")
            ad = ad.replace("'","").replace("[","").replace("]","")
            mdp = mdp.replace("'","").replace("[","").replace("]","")
            mdr = mdr.replace("'","").replace("[","").replace("]","")
            print(str(count)+ ") " + method +":" \
                "| PCD:"+ str(pcd), \
                "| LC:"+ str(lc), \
                "| CRRS:"+ str(crrs), \
                "| CRSR:"+ str(crsr), \
                "| SC:"+ str(sc), \
                "| AD:"+ str(ad), \
                "| MDP:"+ str(mdp), \
                "| MDR:"+ str(mdr)    
                )
    else:
        raise Exception("Transaction data not found")
    delay(start_time, end_time)
    return sorted_query_list

def do_consistency(isConsistent):
    '''Subcommand to regester if any inconsistencies are found. Calls client class to do the registering.'''
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    response = client.consistency(isConsistent)
    print("Find Response: {}".format(response))

def do_isConsistent():
    '''Subcommand to show if there is consistency.  Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = [
        tx.split(',')
        for txs in client.isConsistent()
        for tx in txs.decode().split('|')
    ]
    end_time = datetime.datetime.now()
    if query_list is not None:
        for tx_data in query_list:
            result = tx_data
            print(str(result))
    else:
        raise Exception("Transaction data not found")
    delay(start_time, end_time)    

def do_compute(args, client):
    '''Subcommand to Compute the total QoS score of the methods for a given purpose. Calls client class to do the computation.'''
    start_time = datetime.datetime.now()
    response = client.compute(args.purpose, args.m1, args.m2, args.m3, args.m4, args.m5, args.m6, args.m7, args.m8)
    end_time = datetime.datetime.now()
    print("Find Response: {}".format(response))
    delay(start_time, end_time)

def do_rank(args):
    '''Subcommand to show the method ranks of a purpose. Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = []
    for txs in client.rank(args.purpose):
        for tx in txs.decode().split('|'):
            tx_data = tx.split(',')
            query_list.append(tx_data)
    end_time = datetime.datetime.now()
    if query_list is not None:
        count = 0;
        for tx_data in query_list:
            purpose, *m = tx_data
            purpose = purpose.replace("'","").replace("[","").replace("]","")
            trimmed_m = []
            for r in range(len(m)):
                trimmed_m.append(m[r].replace("[","").replace("]","").replace("'","").replace("\"","").replace(")","").replace("(","").strip(" "))
            print("\nRankings for " + purpose +": ")
            count = 1
            for i in range(0, len(trimmed_m), 2):
                print(str(count) + ". " + trimmed_m[i] + " (Score: "+ str(trimmed_m[i+1]) + ")")
                i = i + 2
                count = count + 1
    else:
        raise Exception("Transaction data not found")
    delay(start_time, end_time)

def do_ranks():
    '''Subcommand to show the method ranks of all.  Calls client class to do the showing.'''
    start_time = datetime.datetime.now()
    privkeyfile = _get_private_keyfile(KEY_NAME)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    query_list = []
    for txs in client.ranks():
        for tx in txs.decode().split('|'):
            tx_data = tx.split(',')
            query_list.append(tx_data)
    end_time = datetime.datetime.now()
    if query_list is not None:
        # Sorting query_list based on the purpose name
        sorted_query_list = sorted(query_list, key=lambda x: x[0].split("'")[1])
        count = 0;
        ranks = {}
        for tx_data in sorted_query_list:
            purpose, *m = tx_data
            purpose = purpose.replace("'","").replace("[","").replace("]","")
            trimmed_m = []
            for r in range(len(m)):
                trimmed_m.append(m[r].replace("[","").replace("]","").replace("'","").replace("\"","").replace(")","").replace("(","").strip(" "))
            print("\nRankings for " + purpose +": ")
            count = 1
            position = 1
            equal_position = False
            for i in range(0, len(trimmed_m), 2):
                #v3:print(str(count) + ". " + trimmed_m[i] + " (Score: "+ str(trimmed_m[i+1]) + ")")
                generator_name = trimmed_m[i]
                score = float(trimmed_m[i+1])
                if equal_position == True:
                    print(str(position) + ". " + generator_name)
                else:
                    print(str(count) + ". " + generator_name)
                count = count + 1    
                if i < len(trimmed_m)-2:
                    if score == float(trimmed_m[i+3]):
                        equal_position = True
                    else:
                        equal_position = False
                        position = count
                #The following lines are used only for checking consistency command
                if purpose not in ranks:
                    ranks[purpose] = []
                ranks[purpose].insert(0,(generator_name,score))
    else:
        raise Exception("Transaction data not found")
    for purpose in ranks:
        ranks[purpose] = sorted(ranks[purpose], key=lambda x: (-x[1], x[0]))
    delay(start_time, end_time)
    return ranks

# Converting the parsed values to float
def convert_method_values(pcd, lc, crrs, crsr, sc, ad, mdp, mdr):
    return {
        "PCD": float(pcd) if pcd is not None else pcd,
        "LC": float(lc) if lc is not None else lc,
        "CRRS": float(crrs) if crrs is not None else crrs,
        "CRSR": float(crsr) if crsr is not None else crsr,
        "SC": float(sc) if sc is not None else sc,
        "AD": float(ad) if ad is not None else ad,
        "MDP": float(mdp) if mdp is not None else mdp,
        "MDR": float(mdr) if mdr is not None else mdr
    }

def convert_cw_values(data_utility, data_privacy):
    return {
        "data_utility": float(data_utility),
        "data_privacy": float(data_privacy)
    }

def local_rank_computation(data_utility_metrics, data_privacy, WM_plus, WM_minus, category_weights, data):

    # Define metric categories
    metrics_lower_is_better = ["PCD", "LC", "AD", "MDP", "MDR"]
    metrics_higher_is_better = ["SC"]
    metrics_constant_is_better = ["CRRS", "CRSR"]

    # Create a DataFrame for transformed data based on rankings
    df = pd.DataFrame(data).T

    # Create a DataFrame for transformed data based on rankings
    transformed_data = df.copy()

    # Iterate through metrics and assign rankings based on metric type categories
    for metric in metrics_lower_is_better:
        transformed_data[metric] = df[metric].rank(method='min', ascending=True)

    for metric in metrics_higher_is_better:
        transformed_data[metric] = df[metric].rank(method='min', ascending=False)

    constant = 1.0  # Constant for "closer to a constant is better" metrics
    for metric in metrics_constant_is_better:
        transformed_data[metric] = abs(df[metric] - constant).rank(method='min', ascending=True)

    inverse_transformed_data = transformed_data.apply(lambda x: x.max() + 1 - x)

    E_plus_matrices = {}
    E_minus_matrices = {}
    
    # Iterate through purposes and store E+ and E- matrices
    for purpose, metric_weights in WM_plus.items():
        E_plus_matrices[purpose] = inverse_transformed_data[list(metric_weights.keys())]
    for purpose, metric_weights in WM_minus.items():
        E_minus_matrices[purpose] = inverse_transformed_data[list(metric_weights.keys())]

    technique_names = ["im", "bn", "mpom", "clgp", "mc_medgan", "mice_lr", "mice_lr_desc", "mice_dt"]

    def rank_techniques(E_plus_matrix, E_minus_matrix, WM_plus, WM_minus, category_weights, data_utility, data_privacy):
        scores = []
        for i in range(len(E_plus_matrix)):
            desired_score = 0
            undesired_score = 0
            generator_name = E_plus_matrix.index[i] # Get the generator name
            for j, metric in enumerate(E_plus_matrix.columns):
                metric_weight = WM_plus[metric]
                if(metric in data_utility):
                    category_weight = category_weights.get('data_utility', 0.0)
                else:
                    category_weight = category_weights.get('data_privacy', 0.0)
                desired_score += E_plus_matrix.iloc[i, j] * metric_weight * category_weight
            for j, metric in enumerate(E_minus_matrix.columns):
                metric_weight = WM_minus[metric]
                if(metric in data_utility):
                    category_weight = category_weights.get('data_utility', 0.0)
                else:
                    category_weight = category_weights.get('data_privacy', 0.0)
                undesired_score += E_minus_matrix.iloc[i, j] * metric_weight * category_weight
            score = desired_score - undesired_score
            scores.append((generator_name, score))

        ranked_techniques = sorted(scores, key=lambda x: x[1], reverse=True)
        return ranked_techniques

    def rank_all_purposes(E_plus_matrices, E_minus_matrices, WM_plus, WM_minus, category_weights, data_utility_metrics, data_privacy):
        all_rankings = {}
        for purpose, weights in category_weights.items():
            ranked = rank_techniques(E_plus_matrices[purpose], E_minus_matrices[purpose], WM_plus[purpose], WM_minus[purpose], weights, data_utility_metrics, data_privacy)
            all_rankings[purpose] = ranked
        return all_rankings

    # Call rank_all_purposes to compute rankings for all purposes
    rankings = rank_all_purposes(E_plus_matrices, E_minus_matrices, WM_plus, WM_minus, category_weights, data_utility_metrics, data_privacy)
    for purpose in rankings:
        rankings[purpose] = sorted(rankings[purpose], key=lambda x: (-x[1], x[0]))
    return rankings

def read_file_content(file_name, isConsistent, mismatch_list):
    try:
        with open(file_name, "r") as file:
            print(f"Reading the {file_name} file...")
            lines = []
            for line in file:
                if line.strip() == "end":
                    break
                lines.append(line)
    except FileNotFoundError:
        isConsistent = False
        mismatch_list.append(f"File {file_name} not found")
        print(f"Error: The file {file_name} was not found.")
    return lines

def check_method_values(isConsistent, mismatch_list):
    
    print("\nComparing the data scientist's input file with the registered data...")
    # Open and read the content of the input.txt file
    lines = read_file_content("inputs.txt", isConsistent, mismatch_list)

    #checking validation of the commands and their inputs, and parsing the data
    data = {}
    for line in lines:
        try:
            # Use regular expressions to extract relevant information
            match = re.match(r'regmethod (\w+) --pcd (\S+) --lc (\S+) --crrs (\S+) --crsr (\S+) --sc (\S+) --ad (\S+) --mdp (\S+) --mdr (\S+)', line)
            if match:
                method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr = match.groups()
                # Update the dictionary with the parsed data
                data[method] = convert_method_values(pcd, lc, crrs, crsr, sc, ad, mdp, mdr)
            else:
                raise ValueError("Invalid format")
        except ValueError as ve:
            isConsistent = False
            mismatch_list.append("Invalid format of inputs.txt")
            print("Error: The inputs.txt file content does not follow the required format!")

    #reading the ledger and matching it with the file content
    print("Reading the ledger for the registered methods and values:")
    ledger_values = do_methods()
    for item in ledger_values:
        method = item[0].strip(" '[]")
        values = [float(val.replace("[","").replace("]","").replace("'","").replace("\"","").replace(")","").replace("(","").strip(" ")) for val in item[1:]]        
        if method in data:
            if data[method] == dict(zip(["PCD", "LC", "CRRS", "CRSR", "SC", "AD", "MDP", "MDR"], values)):
                print(f"Method {method}: Match")
            else:
                isConsistent = False
                mismatch_list.append(f"Method {method} in inputs.txt") 
                print(f"Method {method}: Mismatch")
        else:
            isConsistent = False
            mismatch_list.append(f"Method names in inputs.txt")
            print(f"Error: {method}: Mismatch in method names")

    return isConsistent, mismatch_list, data

def check_category_weights(isConsistent, mismatch_list):

    print("\nComparing the category weights' file with the registered data...")
    # Open and read the content of the weights.txt file
    lines = read_file_content("weights.txt", isConsistent, mismatch_list)

    #checking validation of the commands and their inputs, and parsing the category weights
    category_weights = {}
    for line in lines:
        try:
            # Use regular expressions to extract relevant information
            match = re.match(r'category_weight (\w+) --data_utility (\S+) --data_privacy (\S+)', line)
            if match:
                purpose, data_utility, data_privacy = match.groups()
                # Update the dictionary with the parsed data
                category_weights[purpose] = convert_cw_values(data_utility, data_privacy)
            else:
                raise ValueError("Invalid format")
        except ValueError as ve:
            isConsistent = False
            mismatch_list.append("Invalid format of weights.txt")
            print("Error: The weights.txt file content does not follow the required format!")

    #reading the ledger and matching it with the file content
    print("Reading the ledger for the registered category weights:")
    ledger_values = do_cws()
    for item in ledger_values:
        purpose = item[0].strip(" '[]")
        values = [float(val.replace("[","").replace("]","").replace("'","").replace("\"","").replace(")","").replace("(","").strip(" ")) for val in item[1:]]        
        if purpose in category_weights:
            if category_weights[purpose] == dict(zip(["data_utility", "data_privacy"], values)):
                print(f"Category weights of {purpose}: Match")
            else:
                isConsistent = False
                mismatch_list.append(f"Category weights of {purpose}: Mismatch")
                print(f"Error: Category weights of {purpose} in weights.txt")
        else:
            isConsistent = False
            mismatch_list.append(f"Purpose names in weights.txt")
            print(f"Error: {purpose}: Mismatch in purpose names")

    return isConsistent, mismatch_list, category_weights

def check_wm_plus(isConsistent, mismatch_list):

    print("\nComparing the WM_plus file with the registered data...")
    # Open and read the content of the WM_plus.txt file
    lines = read_file_content("WM_plus.txt", isConsistent, mismatch_list)

    #checking validation of the commands and their inputs, and parsing the WM_plus
    WM_plus = {}
    for line in lines:
        try:
            # Use regular expressions to extract relevant information
            match = re.match(r'WM_plus (\w+)(?: --PCD (\S+))?(?: --LC (\S+))?(?: --CRRS (\S+))?(?: --CRSR (\S+))?(?: --SC (\S+))?(?: --AD (\S+))?(?: --MDP (\S+))?(?: --MDR (\S+))?', line)
            if match:
                purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr = match.groups()
                # Update the dictionary with the parsed data
                WM_plus[purpose] = convert_method_values(pcd, lc, crrs, crsr, sc, ad, mdp, mdr)
            else:
                raise ValueError("Invalid format")
        except ValueError as ve:
            isConsistent = False
            mismatch_list.append("Invalid format of WM_plus.txt")
            print("Error: The WM_plus.txt file content does not follow the required format!")

    #reading the ledger and matching it with the file content
    print("Reading the ledger for the registered WM_plus:")
    ledger_values = do_wmps()
    for item in ledger_values:
        purpose = item[0].strip(" '[]")
        values = []
        for val in item[1:]:
            cleaned_val = val.replace("[", "").replace("]", "").replace("'", "").replace("\"", "").replace(")", "").replace("(", "").strip(" ")
            if cleaned_val != 'None':
                try:
                    values.append(float(cleaned_val))
                except ValueError:
                    print(f"Error converting '{cleaned_val}' to float.")
            else:
                values.append(None)
        if purpose in WM_plus:
            if WM_plus[purpose] == dict(zip(["PCD", "LC", "CRRS", "CRSR", "SC", "AD", "MDP", "MDR"], values)):
                print(f"WM_plus of {purpose}: Match")
            else:
                isConsistent = False
                mismatch_list.append(f"WM_plus of {purpose}: Mismatch")
                print(f"Error: WM_plus of {purpose} in WM_plus.txt")
        else:
            isConsistent = False
            mismatch_list.append(f"Purpose names in WM_plus.txt")
            print(f"Error: {purpose}: Mismatch in purpose names")
    
    # Create a new dictionary with None values omitted
    WM_plus_no_none = {key: {k: v for k, v in subdict.items() if v is not None} for key, subdict in WM_plus.items()}

    return isConsistent, mismatch_list, WM_plus_no_none 

def check_wm_minus(isConsistent, mismatch_list):

    print("\nComparing the WM_minus file with the registered data...")
    # Open and read the content of the WM_minus.txt file
    lines = read_file_content("WM_minus.txt", isConsistent, mismatch_list)

    #checking validation of the commands and their inputs, and parsing the WM_minus
    WM_minus = {}
    for line in lines:
        try:
            # Use regular expressions to extract relevant information
            match = re.match(r'WM_minus (\w+)(?: --PCD (\S+))?(?: --LC (\S+))?(?: --CRRS (\S+))?(?: --CRSR (\S+))?(?: --SC (\S+))?(?: --AD (\S+))?(?: --MDP (\S+))?(?: --MDR (\S+))?', line)
            if match:
                purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr = match.groups()
                # Update the dictionary with the parsed data
                WM_minus[purpose] = convert_method_values(pcd, lc, crrs, crsr, sc, ad, mdp, mdr)
            else:
                raise ValueError("Invalid format")
        except ValueError as ve:
            isConsistent = False
            mismatch_list.append("Invalid format of WM_minus.txt")
            print("Error: The WM_minus.txt file content does not follow the required format!")

    #reading the ledger and matching it with the file content
    print("Reading the ledger for the registered WM_minus:")
    ledger_values = do_wmms()
    for item in ledger_values:
        purpose = item[0].strip(" '[]")
        values = []
        for val in item[1:]:
            cleaned_val = val.replace("[", "").replace("]", "").replace("'", "").replace("\"", "").replace(")", "").replace("(", "").strip(" ")
            if cleaned_val != 'None':
                try:
                    values.append(float(cleaned_val))
                except ValueError:
                    print(f"Error converting '{cleaned_val}' to float.")
            else:
                values.append(None)
        if purpose in WM_minus:
            if WM_minus[purpose] == dict(zip(["PCD", "LC", "CRRS", "CRSR", "SC", "AD", "MDP", "MDR"], values)):
                print(f"WM_minus of {purpose}: Match")
            else:
                isConsistent = False
                mismatch_list.append(f"WM_minus of {purpose}: Mismatch")
                print(f"Error: WM_minus of {purpose} in WM_minus.txt")
        else:
            isConsistent = False
            mismatch_list.append(f"Purpose names in WM_minus.txt")
            print(f"Error: {purpose}: Mismatch in purpose names")

    # Create a new dictionary with None values omitted
    WM_minus_no_none = {key: {k: v for k, v in subdict.items() if v is not None} for key, subdict in WM_minus.items()}

    return isConsistent, mismatch_list, WM_minus_no_none

def check_qi_values(isConsistent, mismatch_list):

    print("\nComparing the qi.txt file with the registered data...")
    # Open and read the content of the qi.txt file
    lines = read_file_content("qi.txt", isConsistent, mismatch_list)

    # Check if the first line starts with "data_utility"
    data_utility = []
    data_privacy = []
    if lines[0].strip().startswith("data_utility"):
        # Extract values from the first line (excluding the "data_utility" part)
        utility_values = lines[0].strip().split()[1:]
        data_utility = utility_values
    else:
        isConsistent = False
        mismatch_list.append("Invalid format of qi.txt")
        print("Error: The qi.txt file content does not follow the required format!")

    # Check if the second line starts with "data_privacy"
    if lines[1].strip().startswith("data_privacy"):
        # Extract values from the second line (excluding the "data_privacy" part)
        privacy_values = lines[1].strip().split()[1:]
        data_privacy = privacy_values
    else:
        isConsistent = False
        mismatch_list.append("Invalid format of qi.txt")
        print("Error: The qi.txt file content does not follow the required format!")

    #reading the ledger and matching it with the file content
    print("Reading the ledger for the registered qis and values:")
    ledger_values = do_qis()
    ledger_row = 0
    for item in ledger_values:
        values = [val.replace("[","").replace("]","").replace("'","").replace("\"","").replace(")","").replace("(","").strip(" ") for val in item]
        if ledger_row == 0:
            if values == privacy_values:
                print(f"Data privacy values: Match")
            else:
                isConsistent = False
                mismatch_list.append(f"Data privacy mismatch in qi.txt") 
                print(f"Data privacy values: Mismatch")
        if ledger_row == 1:
            if values == utility_values:
                print(f"Data utility values: Match")
            else:
                isConsistent = False
                mismatch_list.append(f"Data utility mismatch in qi.txt") 
                print(f"Data utility values: Mismatch")
        ledger_row = ledger_row + 1            

    return isConsistent, mismatch_list, data_utility, data_privacy

def check_ranks(isConsistent, mismatch_list, data_utility, data_privacy, WM_plus, WM_minus, category_weights, data):

    print("\nComparing the registered ranks with the self computation based on the given files...")
    # Open and read the content of the compute.txt file
    lines = read_file_content("compute.txt", isConsistent, mismatch_list)

    # Check if each line starts with "compute"
    compute_dict = {}
    for line in lines:
        if line.strip().startswith("compute"):
            # Extract values from the line (excluding the "compute" part)
            compute_values = line.strip().split()[1:]
            compute_dict[compute_values[0]] = compute_values[1:]
        else:
            isConsistent = False
            mismatch_list.append("Invalid format of compute.txt")
            print("Error: The compute.txt file content does not follow the required format!")

    print("Computing the rankings based on the given files...")
    local_ranks = local_rank_computation(data_utility, data_privacy, WM_plus, WM_minus, category_weights, data)
    print("rankings based on the given files:",local_ranks)

    print("Reading the ledger for the registered ranks:")
    ledger_ranks = do_ranks()

    print("ledger ranks:", ledger_ranks)

    # Check if the ranking from the ledger and the one from the computation based on the files are identical
    if local_ranks == ledger_ranks:
        print("Rankings from the ledger and from the computation base on the files: Match")
    else:
        print("Rankings from the ledger and from the computation base on the files: Mismatch")
        # Find the differences
        differences = {key: [(a, b) for a, b in zip(local_ranks[key], ledger_ranks[key]) if a != b] for key in local_ranks if key in ledger_ranks}
        print("Differences:")
        print(differences)
        isConsistent = False
        mismatch_list.append(f"rankings mismatch") 

    return isConsistent, mismatch_list

def do_auditing(args):
    '''Subcommand to audit the registered data into the ledger'''
    start_time = datetime.datetime.now()
    isConsistent = True
    mismatch_list = []
    isConsistent, mismatch_list, data = check_method_values(isConsistent, mismatch_list)
    isConsistent, mismatch_list, category_weights = check_category_weights(isConsistent, mismatch_list)
    isConsistent, mismatch_list, WM_plus = check_wm_plus(isConsistent, mismatch_list)
    isConsistent, mismatch_list, WM_minus = check_wm_minus(isConsistent, mismatch_list)
    isConsistent, mismatch_list, data_utility, data_privacy = check_qi_values(isConsistent, mismatch_list)
    isConsistent, mismatch_list = check_ranks(isConsistent, mismatch_list, data_utility, data_privacy, WM_plus, WM_minus, category_weights, data)

    print("\n***Auditing process completed***")
    if(isConsistent):
        print("***All the registered data in the ledger are consistent with the given files.***")
    else:
        print("***Mismatch found between the registered data in the ledger and the given files:")
        for mismatch in mismatch_list:
            print(mismatch)

    print("\nRegistering the consistency result to the ledger:")
    do_consistency(isConsistent)
    end_time = datetime.datetime.now()
    delay(start_time, end_time)

def read_file_commands(args):
    #Every user can run the compute file and the "qos" command, but a valid key is needed for running other commands 
    if args.command == 'qos':
        privkeyfile = _get_private_keyfile(KEY_NAME)
    else:
        privkeyfile = _get_private_keyfile(args.key)
    client = synthrankClient(base_url=DEFAULT_URL, key_file=privkeyfile)
    command_file = open(args.filepath, 'r')
    while True:
        # Get next line from file
        line = command_file.readline()
        line_args = line.split()
        if line_args[0] == "end":
            break
        parser = create_parser(os.path.basename(sys.argv[0]))
        line_args = parser.parse_args(line_args)
        function_dispatcher(line_args, client)
    command_file.close()

def delay(start_time, end_time):
    delta = end_time - start_time
    print("\nThe time between sending the command and getting the ack: "+ str(delta.total_seconds())+" seconds")

def function_dispatcher(args, client=None):
    if args.command == 'method':
        read_file_commands(args)
    elif args.command == 'qos':
        read_file_commands(args)
    elif args.command == 'qi':
        read_file_commands(args)
    elif args.command == 'audit':
        do_auditing(args)
    elif args.command == 'data_utility':
        do_data_utility(args, client)
    elif args.command == 'data_privacy':
        do_data_privacy(args, client)
    elif args.command == 'qis':
        do_qis()
    elif args.command == 'cw':
        read_file_commands(args)
    elif args.command == 'category_weight':
        do_category_weight(args, client)
    elif args.command == 'cws':
        do_cws()
    elif args.command == 'wmp':
        read_file_commands(args)
    elif args.command == 'WM_plus':
        do_wm_plus(args, client)
    elif args.command == 'wmps':
        do_wmps()
    elif args.command == 'wmm':
        read_file_commands(args)
    elif args.command == 'WM_minus':
        do_wm_minus(args, client)
    elif args.command == 'wmms':
        do_wmms()                
    elif args.command == 'regmethod':
        do_regmethod(args, client)
    elif args.command == 'delmethod':
        do_delmethod(args)
    elif args.command == 'methods':
        do_methods()
    elif args.command == 'compute':
        do_compute(args, client)
    elif args.command == 'rank':
        do_rank(args)
    elif args.command == 'ranks':
        do_ranks()
    elif args.command == 'isConsistent':
        do_isConsistent()                                       	
    else:
        raise Exception("Invalid command: {}".format(args.command))

def main(prog_name=os.path.basename(sys.argv[0]), args=None):
    '''Entry point function for the client CLI.'''
    try:
        if args is None:
            args = sys.argv[1:]
        parser = create_parser(prog_name)
        args = parser.parse_args(args)
        verbose_level = 0
        setup_loggers(verbose_level=verbose_level)
        function_dispatcher(args)

    except KeyboardInterrupt:
        pass
    except SystemExit as err:
        raise err
    except BaseException as err:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()


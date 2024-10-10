#!/usr/bin/env python3

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

# developed by Mohammad H. T.

'''
synthrankTransactionHandler class interfaces for synthrank Transaction Family.
'''

import traceback
import sys
import hashlib
import logging
import random
import string
import os.path
import json

from sawtooth_sdk.processor.handler import TransactionHandler
from sawtooth_sdk.processor.exceptions import InvalidTransaction
from sawtooth_sdk.processor.exceptions import InternalError
from sawtooth_sdk.processor.core import TransactionProcessor
from pathlib import Path
from collections import ChainMap
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# hard-coded for simplicity (otherwise get the URL from the args in main):
DEFAULT_URL = 'tcp://localhost:4004'
# For Docker:
#DEFAULT_URL = 'tcp://validator:4004'
 
LOGGER = logging.getLogger(__name__)

FAMILY_NAME = "synthrank"
# TF Prefix is first 6 characters of SHA-512("synthrank")
METHODS_NS = "methods"
SCENARIOS_NS = "scenarios"
RANKS_NS = "rankings"
QIS_NS = "indicators"
DATA_UTILITY_NS = "utility"
DATA_PRIVACY_NS = "privacy"
CWS_NS = "weights"
WMPS_NS = "plus"
WMMS_NS = "minus"
CONSISTENCY_NS = "consistency"

scaler = MinMaxScaler()

def _hash(data):
    '''Compute the SHA-512 hash and return the result as hex characters.'''
    return hashlib.sha512(data).hexdigest()

def _get_consistency_address():
    return _hash(CONSISTENCY_NS.encode('utf-8'))[0:70]

def _get_prefix_qis():
    return _hash(QIS_NS.encode('utf-8'))[0:6]

def _get_data_utility_address():
    return _get_prefix_qis() + \
        _hash(DATA_UTILITY_NS.encode('utf-8'))[0:64]

def _get_data_privacy_address():
    return _get_prefix_qis() + \
        _hash(DATA_PRIVACY_NS.encode('utf-8'))[0:64]

def _get_prefix_cws():
    return _hash(CWS_NS.encode('utf-8'))[0:6]

def _get_category_weight_address(id):
    return _get_prefix_cws() + \
        _hash(id.encode('utf-8'))[0:64]

def _get_prefix_wmps():
    return _hash(WMPS_NS.encode('utf-8'))[0:6]

def _get_wm_plus_address(id):
    return _get_prefix_wmps() + \
        _hash(id.encode('utf-8'))[0:64]

def _get_prefix_wmms():
    return _hash(WMMS_NS.encode('utf-8'))[0:6]

def _get_wm_minus_address(id):
    return _get_prefix_wmms() + \
        _hash(id.encode('utf-8'))[0:64]

def _get_prefix_methods():
    return _hash(METHODS_NS.encode('utf-8'))[0:6]

def _get_method_address(from_key,id):
    '''
    Return the address of a synthrank method from the synthrank TF.

    The address is the first 6 hex characters from the hash SHA-512(NS name),
    plus the result of the hash SHA-512(method).
    '''
    return _hash(METHODS_NS.encode('utf-8'))[0:6] + \
                 _hash(id.encode('utf-8'))[0:64]

def _get_scenario_address(from_key,id):
    '''
    Return the address of a synthrank scenario from the synthrank TF.

    The address is the first 6 hex characters from the hash SHA-512(NS name),
    plus the result of the hash SHA-512(scenario).
    '''
    return _hash(SCENARIOS_NS.encode('utf-8'))[0:6] + \
                 _hash(id.encode('utf-8'))[0:64]

def _get_rank_address(id):
    '''
    Return the address of a scenario's method rankings from the synthrank TF.

    The address is the first 6 hex characters from the hash SHA-512(NS name),
    plus the result of the hash SHA-512(scenario).
    '''
    return _hash(RANKS_NS.encode('utf-8'))[0:6] + \
                 _hash(id.encode('utf-8'))[0:64]

def determine_metric_category(metric_name, lower_is_better_metrics, higher_is_better_metrics):
    if metric_name in lower_is_better_metrics:
        return "lower_is_better"
    elif metric_name in higher_is_better_metrics:
        return "higher_is_better"
    else:
        return "unknown"

# Safe log transformation for "higher is better" metrics
def safe_log_transform_higher_is_better_undesirable(metric_value):
    return np.log1p(metric_value) + 1

# Transformation for "lower is better" metrics
def transform_lower_is_better(metric_value):
    # Inverting the metric value to reflect its desirability
    return 1 / np.abs(metric_value) + 1

def transform_negative_values(metric_value, metric_min_value):
    # Adding a constant to shift the negative values to positive domain and then applying the inverse transformation
    shifted_value = metric_value + np.abs(metric_min_value) + 1
    return 1 / shifted_value

def transform_lower_is_better_privacy(metric_value):
    return 1 / (metric_value + 1e-6)

def inverse_transform_negative_values(metric_value, min_value):
    # Adding a constant to ensure non-zero values and then applying the inverse transformation
    shifted_value = metric_value + abs(min_value) + 1
    return 1 / shifted_value

def generate_E_matrices(purpose, methods, data, transformed_lower_data_desired, transformed_higher_data_undesired_log_safe, WM_plus, WM_minus, transformed_lower_data_undesired_negative_adjusted):

    E_plus_matrices = {}
    E_minus_matrices = {}

    E_plus = []
    E_minus = []

    for generator in data.keys():
        E_plus_row = []
        E_minus_row = []

        for metric in WM_plus:
            if metric in transformed_lower_data_desired[generator]:
                E_plus_row.append(transformed_lower_data_desired[generator][metric])
            else:
                E_plus_row.append(data[generator][metric])

        for metric in WM_minus:
            if metric in transformed_higher_data_undesired_log_safe[generator]:
                E_minus_row.append(transformed_higher_data_undesired_log_safe[generator][metric])
            elif metric in transformed_lower_data_undesired_negative_adjusted[generator]:
                # Use adjusted transformation for negative lower-is-better metrics
                E_minus_row.append(transformed_lower_data_undesired_negative_adjusted[generator][metric])
            else:
                E_minus_row.append(data[generator][metric])

        E_plus.append(E_plus_row)
        E_minus.append(E_minus_row)

    E_plus_matrices[purpose] = E_plus
    E_minus_matrices[purpose] = E_minus

    return E_plus_matrices, E_minus_matrices

def normalize_score(score, min_score, max_score, min_normalized=1, max_normalized=5):
    if max_score == min_score:
            return min_normalized  # Avoid division by zero
    return ((score - min_score) / (max_score - min_score)) * (max_normalized - min_normalized) + min_normalized

def rank_purpose(context, purpose, E_plus, E_minus, WM_plus, WM_minus, category_weights, methods, data_utility):
    
    scores = []

    for i in range(len(E_plus[purpose])):
        score = 0
        for j, metric in enumerate(WM_plus):
            metric_weight = WM_plus[metric]
            category_weight = category_weights[0 if metric in data_utility else 1]
            score += E_plus[purpose][i][j] * metric_weight * category_weight

        for j, metric in enumerate(WM_minus):
            metric_weight = WM_minus[metric]
            category_weight = category_weights[0 if metric in data_utility else 1]
            score -= E_minus[purpose][i][j] * metric_weight * category_weight

        scores.append((methods[i], score))
        LOGGER.info("method %s score: %s", methods[i], score) 

    min_score = min(scores, key=lambda x: x[1])[1]
    max_score = max(scores, key=lambda x: x[1])[1]

    scaled_scores = [(tech, normalize_score(score, min_score, max_score)) for tech, score in scores]
    ranked_techniques = sorted(scaled_scores, key=lambda x: x[1], reverse=True)
    return ranked_techniques                                

class synthrankTransactionHandler(TransactionHandler):
    '''
    Transaction Processor class for the synthrank Transaction Family.

    This TP communicates with the Validator using the accept/get/set functions
    This implements functions to "find".
    '''
    def __init__(self, namespace_prefix):
        '''Initialize the transaction handler class.

           This is setting the "synthrank" TF namespace prefix.
        '''
        self._namespace_prefix = namespace_prefix

    @property
    def family_name(self):
        '''Return Transaction Family name string.'''
        return FAMILY_NAME

    @property
    def family_versions(self):
        '''Return Transaction Family version string.'''
        return ['1.0']

    @property
    def namespaces(self):
        '''Return Transaction Family namespace 6-character prefix.'''
        return [self._namespace_prefix]

    def apply(self, transaction, context):
        '''This implements the apply function for the TransactionHandler class.

           The apply function does most of the work for this class by
           processing a transaction for the synthrank transaction family.
        '''

        # Get the payload and extract the synthrank-specific information.
        # It has already been converted from Base64, but needs deserializing.
        # It was serialized with CSV: action, value
        header = transaction.header
        payload_list = transaction.payload.decode().split(",")
        action = payload_list[0]
        if action == "data_utility":
            data_utility = payload_list[1:]
            LOGGER.info("data utility = %s.", data_utility)
        elif action == "data_privacy":
            data_privacy = payload_list[1:]
            LOGGER.info("data privacy = %s.", data_privacy)
        elif action == "category_weight":
            purpose = payload_list[1]
            LOGGER.info("purpose = %s.", purpose)   
            data_utility = payload_list[2]
            LOGGER.info("data_utility = %s.", data_utility)
            data_privacy = payload_list[3]
            LOGGER.info("data_privacy = %s.", data_privacy)
        elif action == "WM_plus":
            purpose = payload_list[1]
            LOGGER.info("purpose = %s.", purpose)   
            pcd = payload_list[2]
            LOGGER.info("pcd = %s.", pcd)
            lc = payload_list[3]
            LOGGER.info("lc = %s.", lc)
            crrs = payload_list[4]
            LOGGER.info("crrs = %s.", crrs)
            crsr = payload_list[5]
            LOGGER.info("crsr = %s.", crsr)
            sc = payload_list[6]
            LOGGER.info("sc = %s.", sc)
            ad = payload_list[7]
            LOGGER.info("ad = %s.", ad)
            mdp = payload_list[8]
            LOGGER.info("mdp = %s.", mdp)
            mdr = payload_list[9]
            LOGGER.info("mdr = %s.", mdr)
        elif action == "WM_minus":
            purpose = payload_list[1]
            LOGGER.info("purpose = %s.", purpose)   
            pcd = payload_list[2]
            LOGGER.info("pcd = %s.", pcd)
            lc = payload_list[3]
            LOGGER.info("lc = %s.", lc)
            crrs = payload_list[4]
            LOGGER.info("crrs = %s.", crrs)
            crsr = payload_list[5]
            LOGGER.info("crsr = %s.", crsr)
            sc = payload_list[6]
            LOGGER.info("sc = %s.", sc)
            ad = payload_list[7]
            LOGGER.info("ad = %s.", ad)
            mdp = payload_list[8]
            LOGGER.info("mdp = %s.", mdp)
            mdr = payload_list[9]
            LOGGER.info("mdr = %s.", mdr)    
        elif action == "regmethod":
            method = payload_list[1]
            LOGGER.info("method = %s.", method)   
            pcd = payload_list[2]
            LOGGER.info("pcd = %s.", pcd)
            lc = payload_list[3]
            LOGGER.info("lc = %s.", lc)
            crrs = payload_list[4]
            LOGGER.info("crrs = %s.", crrs)
            crsr = payload_list[5]
            LOGGER.info("crsr = %s.", crsr)
            sc = payload_list[6]
            LOGGER.info("sc = %s.", sc)
            ad = payload_list[7]
            LOGGER.info("ad = %s.", ad)
            mdp = payload_list[8]
            LOGGER.info("mdp = %s.", mdp)
            mdr = payload_list[9]
            LOGGER.info("mdr = %s.", mdr)
        elif action == "delmethod":
            method = payload_list[1]
            LOGGER.info("method = %s.", method)
        elif action == "regscenario":
            scenario = payload_list[1]
            LOGGER.info("scenario = %s.", scenario)   
            pcd = payload_list[2]
            LOGGER.info("pcd = %s.", pcd)
            lc = payload_list[3]
            LOGGER.info("lc = %s.", lc)
            crrs = payload_list[4]
            LOGGER.info("crrs = %s.", crrs)
            crsr = payload_list[5]
            LOGGER.info("crsr = %s.", crsr)
            sc = payload_list[6]
            LOGGER.info("sc = %s.", sc)
            ad = payload_list[7]
            LOGGER.info("ad = %s.", ad)
            mdp = payload_list[8]
            LOGGER.info("mdp = %s.", mdp)
            mdr = payload_list[9]
            LOGGER.info("mdr = %s.", mdr)
        elif action == "delscenario":
            scenario = payload_list[1]
            LOGGER.info("scenario = %s.", scenario)
        elif action == "compute":
            purpose = payload_list[1]
            LOGGER.info("purpose = %s.", purpose)
            m1 = payload_list[2]
            LOGGER.info("method1 = %s.", m1)
            m2 = payload_list[3]
            LOGGER.info("method2 = %s.", m2)
            m3 = payload_list[4]
            LOGGER.info("method3 = %s.", m3)
            m4 = payload_list[5]
            LOGGER.info("method4 = %s.", m4)
            m5 = payload_list[6]
            LOGGER.info("method5 = %s.", m5)
            m6 = payload_list[7]
            LOGGER.info("method6 = %s.", m6)
            m7 = payload_list[8]
            LOGGER.info("method7 = %s.", m7)
            m8 = payload_list[9]
            LOGGER.info("method8 = %s.", m8)
        elif action == "consistency":
            approval = payload_list[1]
            LOGGER.info("consistency = %s.", approval)            

        # Get the signer's public key, sent in the header from the client.
        from_key = header.signer_public_key

        # Perform the action.
        if action == "data_utility":
            self._make_data_utility(context, data_utility)
        elif action == "data_privacy":
            self._make_data_privacy(context, data_privacy)
        elif action == "category_weight":
            self._make_category_weight(context, purpose, data_utility, data_privacy)
        elif action == "WM_plus":
            self._make_wm_plus(context, purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr)
        elif action == "WM_minus":
            self._make_wm_minus(context, purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr)      
        elif action == "regmethod":
            self._make_regmethod(context, method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr, from_key)
        elif action == "delmethod":
            self._make_delmethod(context, method, from_key)
        elif action == "regscenario":
            self._make_regscenario(context, scenario, pcd, lc, crrs, crsr, sc, ad, mdp, mdr, from_key)
        elif action == "delscenario":
            self._make_delscenario(context, scenario, from_key)
        elif action == "compute":
            self._make_compute(context, purpose, m1, m2, m3, m4, m5, m6, m7, m8, from_key)
        elif action == "consistency":
            self._make_consistency(context, approval)           
        else:
            LOGGER.info("Unhandled action. Action should be data_utility or data_privacy or category_weight or WM_plus or WM_minus or regmethod or compute")

    @classmethod
    def _make_consistency(cls, context, approval):
        '''populate the ledger with QIs'''
        consistency_address = _get_consistency_address()
        LOGGER.info('Got the consistency address %s.', consistency_address)                        
        state_data = str(approval).encode('utf-8')
        context.set_state({consistency_address: state_data})

    @classmethod
    def _make_data_utility(cls, context, data_utility):
        '''populate the ledger with QIs'''
        data_utility_address = _get_data_utility_address()
        LOGGER.info('Got the data utility address %s.', data_utility_address)                        
        state_data = str(data_utility).encode('utf-8')
        context.set_state({data_utility_address: state_data})

    @classmethod
    def _make_data_privacy(cls, context, data_privacy):
        '''populate the ledger with QIs'''
        data_privacy_address = _get_data_privacy_address()
        LOGGER.info('Got the data privacy address %s.', data_privacy_address)                        
        state_data = str(data_privacy).encode('utf-8')
        context.set_state({data_privacy_address: state_data})

    @classmethod
    def _make_category_weight(cls, context, purpose, data_utility, data_privacy):
        '''populate the ledger with category weights'''
        category_weight_address = _get_category_weight_address(purpose)
        LOGGER.info('Got the category weight address %s.', category_weight_address)
        p = [purpose, data_utility, data_privacy]                        
        state_data = str(p).encode('utf-8')
        context.set_state({category_weight_address: state_data})

    @classmethod
    def _make_wm_plus(cls, context, purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr):
        '''populate the ledger with category weights'''
        wm_plus_address = _get_wm_plus_address(purpose)
        LOGGER.info('Got the wm plus address %s.', wm_plus_address)
        wmp = [purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr]                        
        state_data = str(wmp).encode('utf-8')
        context.set_state({wm_plus_address: state_data})

    @classmethod
    def _make_wm_minus(cls, context, purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr):
        '''populate the ledger with category weights'''
        wm_minus_address = _get_wm_minus_address(purpose)
        LOGGER.info('Got the wm minus address %s.', wm_minus_address)
        wmm = [purpose, pcd, lc, crrs, crsr, sc, ad, mdp, mdr]                        
        state_data = str(wmm).encode('utf-8')
        context.set_state({wm_minus_address: state_data})

    @classmethod
    def _make_regmethod(cls, context, method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr, from_key):
        '''populate the ledger with a method and its values.'''
        method_address = _get_method_address(from_key,method)
        LOGGER.info('Got the key %s and the method address %s.',from_key, method_address)                        
        m = [method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr]
        state_data = str(m).encode('utf-8')
        context.set_state({method_address: state_data})

    @classmethod
    def _make_delmethod(cls, context, method, from_key):
        query_address = _get_method_address(from_key,method)
        LOGGER.info('Got the key %s and the method address %s.',from_key, query_address)
        context.delete_state([query_address])

    @classmethod
    def _make_regscenario(cls, context, scenario, pcd, lc, crrs, crsr, sc, ad, mdp, mdr, from_key):
        '''populate the ledger with a scenario and its weights.'''
        scenario_address = _get_scenario_address(from_key,scenario)
        LOGGER.info('Got the key %s and the scenario address %s.',from_key, scenario_address)                        
        s = [scenario, pcd, lc, crrs, crsr, sc, ad, mdp, mdr]
        state_data = str(s).encode('utf-8')
        context.set_state({scenario_address: state_data})

    @classmethod
    def _make_delscenario(cls, context, scenario, from_key):
        query_address = _get_scenario_address(from_key,scenario)
        LOGGER.info('Got the key %s and the scenario address %s.',from_key, query_address)
        context.delete_state([query_address])

    @classmethod
    def _make_compute(cls, context, purpose, m1, m2, m3, m4, m5, m6, m7, m8, from_key):
        '''populate the ledger with a purpose's method rankings.'''

        metrics = ["PCD", "LC", "CRRS", "CRSR", "SC", "AD", "MDP", "MDR"]
        lower_is_better_metrics = ["PCD", "LC", "AD", "MDP", "MDR"]
        higher_is_better_metrics = ["CRRS", "CRSR", "SC"]

        data_utility_address = _get_data_utility_address()
        LOGGER.info('Got the data utility address %s.', data_utility_address)
        state_entries_data_utility = context.get_state([data_utility_address])
        d_utility = state_entries_data_utility[0].data.decode().split(',') #d_utility = ["PCD", "CRRS", "CRSR", "LC", "SC"]
        LOGGER.info('Got the data utility: %s', d_utility)
        data_utility = np.array([val.replace("'","").replace("[","").replace("]","").replace("\\","").replace("\"","").strip() for val in d_utility]).tolist()
        LOGGER.info('Got the trimmed data utility: %s', data_utility) #data_utility = ["PCD", "CRRS", "CRSR", "LC", "SC"]

        data_privacy_address = _get_data_privacy_address()
        LOGGER.info('Got the data privacy address %s.', data_privacy_address)
        state_entries_data_privacy = context.get_state([data_privacy_address])
        d_privacy = state_entries_data_privacy[0].data.decode().split(',') #d_privacy = ["data_privacy", "AD", "MDP", "MDR"]
        LOGGER.info('Got the data privacy: %s', d_privacy)
        data_privacy = np.array([val.replace("'","").replace("[","").replace("]","").replace("\\","").replace("\"","").strip() for val in d_privacy]).tolist()
        LOGGER.info('Got the trimmed data privacy: %s', data_privacy) #data_privacy = ["AD", "MDP", "MDR"]

        category_weight_address = _get_category_weight_address(purpose)
        LOGGER.info('Got the category weight address %s.', category_weight_address)
        state_entries_category_weight = context.get_state([category_weight_address])
        c_weights = state_entries_category_weight[0].data.decode().split(',') #c_weights = [PurposeA, data_utility, data_privacy]
        LOGGER.info('Got the category weights: %s', c_weights[1:])
        # Convert strings to floats, handling potential issues
        category_weights = np.array([float(val.strip().strip("]").strip("'")) for val in c_weights[1:]]).tolist()
        LOGGER.info('Got the float category weights: %s', category_weights) #category_weights = [0.7 , 0.3]

        wm_plus_address = _get_wm_plus_address(purpose)
        LOGGER.info('Got the wm plus address %s.', wm_plus_address)
        state_entries_wm_plus = context.get_state([wm_plus_address])
        wmplus_weights = state_entries_wm_plus[0].data.decode().split(',') #wmplus_weights = [PurposeA, pcd, lc, crrs, crsr, sc, ad, mdp, mdr]
        LOGGER.info('Got the wmplus weights: %s', wmplus_weights[1:])
        # Convert strings to floats, handling potential issues
        # Replace 'None' with -99
        wmplus_weights = [
            float(val.strip().strip("]").strip("'").replace("None", "-99"))
            for val in wmplus_weights[1:]
        ]
        # Convert strings to floats
        WM_plus = np.array(wmplus_weights).tolist()
        WM_plus = {key: value for key, value in zip(metrics, WM_plus) if value != -99.0} #WM_plus {'PCD': 0.4, 'CRSR': 0.6}
        LOGGER.info('Got the float wmplus weights: %s', WM_plus)

        wm_minus_address = _get_wm_minus_address(purpose)
        LOGGER.info('Got the wm minus address %s.', wm_minus_address)
        state_entries_wm_minus = context.get_state([wm_minus_address])
        wmminus_weights = state_entries_wm_minus[0].data.decode().split(',') #wmminus_weights = [PurposeA, pcd, lc, crrs, crsr, sc, ad, mdp, mdr]
        LOGGER.info('Got the wm minus weights: %s', wmminus_weights[1:])
        # Convert strings to floats, handling potential issues
        # Replace 'None' with -99
        wmminus_weights = [
            float(val.strip().strip("]").strip("'").replace("None", "-99"))
            for val in wmminus_weights[1:]
        ]
        # Convert strings to floats
        WM_minus = np.array(wmminus_weights).tolist()
        WM_minus = {key: value for key, value in zip(metrics, WM_minus) if value != -99.0} #WM_minus {'MDP': 0.6, 'MDR': 0.4}
        LOGGER.info('Got the float wm minus weights: %s', WM_minus)

        methods = [m1, m2, m3, m4, m5, m6, m7, m8]
        data_raw = []
        for m in methods:
            method_address = _get_method_address(from_key,m)
            state_entries_method = context.get_state([method_address])
            m_values = state_entries_method[0].data.decode().split(',') #m_values = [method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr]
            method = m_values[0]
            mv = np.array([float(val.strip().strip("]").strip("'")) for val in m_values[1:]]).tolist()
            #LOGGER.info('Got the method values: %s', mv)
            data_raw.append(mv)
        LOGGER.info('Got the raw data values: %s', data_raw)
        data = {}
        for method in methods:
            sub_data = {}
            for metric in metrics:
                sub_data[metric] = data_raw[methods.index(method)][metrics.index(metric)]
            data[method] = sub_data
        LOGGER.info('Got the data dictionary values: %s', data)


        min_values = {metric: min(data[generator][metric] for generator in data if metric in data[generator])
                    for metric in lower_is_better_metrics}
        LOGGER.info("Got the min_values: %s", min_values)

        # Apply transformations
        transformed_higher_data_undesired_log_safe = {}
        transformed_lower_data_desired = {}
        for generator, metrics in data.items():
            transformed_metrics_higher = {}
            transformed_metrics_lower = {}
            for metric, value in metrics.items():
                metric_category = determine_metric_category(metric, lower_is_better_metrics, higher_is_better_metrics)
                if metric_category == "lower_is_better":
                    if value < 0 and metric in min_values:
                        transformed_metrics_lower[metric] = transform_negative_values(value, min_values[metric])
                    if metric == "PCD" or metric == "LC":
                        transformed_metrics_lower[metric] = transform_lower_is_better(value)
                    if metric == "AD" or metric == "MDP" or metric == "MDR":
                        transformed_metrics_lower[metric] = transform_lower_is_better_privacy(value)
                elif metric_category == "higher_is_better":
                    transformed_metrics_higher[metric] = safe_log_transform_higher_is_better_undesirable(value)
                else:
                    transformed_metrics_lower[metric] = value
                    transformed_metrics_higher[metric] = value
            transformed_lower_data_desired[generator] = transformed_metrics_lower
            transformed_higher_data_undesired_log_safe[generator] = transformed_metrics_higher

        # Determine the minimum values for each metric across all generators for negative value handling
        min_values = {metric: min(data[generator][metric] for generator in data if metric in data[generator])
              for metric in lower_is_better_metrics if any(data[generator][metric] < 0 for generator in data)}
        LOGGER.info("Got the min_value: %s", min_values)

        # Apply transformations for negative values in lower_is_better_metrics
        transformed_lower_data_undesired_negative_adjusted = {}
        for generator, metrics in data.items():
            transformed_metrics = {}
            for metric, value in metrics.items():
                if metric in lower_is_better_metrics and value < 0:
                    transformed_metrics[metric] = inverse_transform_negative_values(value, min_values[metric])
                else:
                    transformed_metrics[metric] = value
            transformed_lower_data_undesired_negative_adjusted[generator] = transformed_metrics
        LOGGER.info("Got the transformed_lower_data_undesired_negative_adjusted: %s", transformed_lower_data_undesired_negative_adjusted)                 

        E_plus_matrices, E_minus_matrices = generate_E_matrices(purpose, methods, data, transformed_lower_data_desired, transformed_higher_data_undesired_log_safe, WM_plus, WM_minus, transformed_lower_data_undesired_negative_adjusted)
        LOGGER.info("Got the E_plus_matrices: %s", E_plus_matrices)
        LOGGER.info("Got the E_minus_matrices: %s", E_minus_matrices)

        rankings = rank_purpose(context, purpose, E_plus_matrices, E_minus_matrices, WM_plus, WM_minus, category_weights, methods, data_utility)
        LOGGER.info('Got the ranks  %s.', rankings)

        rank = [purpose, rankings]
        state_data = str(rank).encode('utf-8')
        rank_address = _get_rank_address(purpose)
        LOGGER.info('Got the ranks address  %s.', rank_address)
        context.set_state({rank_address: state_data})     

def main():
    '''Entry-point function for the smartmed Transaction Processor.'''
    try:
        # Setup logging for this class.
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)

        # Register the Transaction Handler and start it.
        processor = TransactionProcessor(url=DEFAULT_URL)
        sw_namespace = _hash(FAMILY_NAME.encode('utf-8'))[0:6]
        handler = synthrankTransactionHandler(sw_namespace)
        processor.add_handler(handler)
        processor.start()
    except KeyboardInterrupt:
        pass
    except SystemExit as err:
        raise err
    except BaseException as err:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

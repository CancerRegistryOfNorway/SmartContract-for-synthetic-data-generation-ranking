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
'''
synthrankClient class interfaces with Sawtooth through the REST API.
It accepts input from a client CLI/GUI/BUI or other interface.
'''
# developed by Mohammad H. Tabatabaei and Narasimha Raghavan 

from builtins import BaseException
import hashlib
import base64
import random
import time
import requests
import yaml

from sawtooth_signing import create_context
from sawtooth_signing import CryptoFactory
from sawtooth_signing import ParseError
from sawtooth_signing.secp256k1 import Secp256k1PrivateKey
from sawtooth_sdk.protobuf.transaction_pb2 import TransactionHeader
from sawtooth_sdk.protobuf.transaction_pb2 import Transaction
from sawtooth_sdk.protobuf.batch_pb2 import BatchList
from sawtooth_sdk.protobuf.batch_pb2 import BatchHeader
from sawtooth_sdk.protobuf.batch_pb2 import Batch

# The Transaction Family Name
FAMILY_NAME = 'synthrank'
# TF Prefix is first 6 characters of SHA-512("synthrank"),
METHODS_NS = "methods"
RANKS_NS = "rankings"
QIS_NS = "indicators"
DATA_UTILITY_NS = "utility"
DATA_PRIVACY_NS = "privacy"
CWS_NS = "weights"
WMPS_NS = "plus"
WMMS_NS = "minus"
CONSISTENCY_NS = "consistency"

def _hash(data):
    return hashlib.sha512(data).hexdigest()

class synthrankClient(object):
    '''Client synthrank class

    Supports "regmethod", "reg_data_utility", "reg_data_privacy", "reg_category_weight", "reg_wm_plus", "reg_wm_minus", "compute" and showing functions.
    '''

    def __init__(self, base_url, key_file):
        '''Initialize the client class.

           This is mainly getting the key pair and computing the address.
        '''
        self._base_url = base_url

        if key_file is None:
            self._signer = None
            return

        try:
            with open(key_file) as key_fd:
                private_key_str = key_fd.read().strip()
        except OSError as err:
            raise Exception(
                'Failed to read private key {}: {}'.format(
                    key_file, str(err)))

        try:
            private_key = Secp256k1PrivateKey.from_hex(private_key_str)
        except ParseError as err:
            raise Exception( \
                'Failed to load private key: {}'.format(str(err)))

        self._signer = CryptoFactory(create_context('secp256k1')) \
            .new_signer(private_key)
        self._public_key = self._signer.get_public_key().as_hex()

    #Getting addresses of the ledger data
        
    def _get_prefix_qis(self):
        return _hash(QIS_NS.encode('utf-8'))[0:6]

    def _get_data_utility_address(self):
        return self._get_prefix_qis() + \
            _hash(DATA_UTILITY_NS.encode('utf-8'))[0:64]

    def _get_data_privacy_address(self):
        return self._get_prefix_qis() + \
            _hash(DATA_PRIVACY_NS.encode('utf-8'))[0:64]

    def _get_prefix_cws(self):
        return _hash(CWS_NS.encode('utf-8'))[0:6]

    def _get_category_weight_address(self, id):
        return self._get_prefix_cws() + \
            _hash(id.encode('utf-8'))[0:64]

    def _get_prefix_wmps(self):
        return _hash(WMPS_NS.encode('utf-8'))[0:6]

    def _get_wm_plus_address(self, id):
        return self._get_prefix_wmps() + \
            _hash(id.encode('utf-8'))[0:64]

    def _get_prefix_wmms(self):
        return _hash(WMMS_NS.encode('utf-8'))[0:6]

    def _get_wm_minus_address(self, id):
        return self._get_prefix_wmms() + \
            _hash(id.encode('utf-8'))[0:64]  

    def _get_prefix_methods(self):
        return _hash(METHODS_NS.encode('utf-8'))[0:6]

    # Method address is 6-char TF prefix + hash of the method  
    def _get_method_address(self, id):
        return self._get_prefix_methods() + \
            _hash(id.encode('utf-8'))[0:64]

    def _get_prefix_ranks(self):
        return _hash(RANKS_NS.encode('utf-8'))[0:6]

    # Purpose address is 6-char TF prefix + hash of the purpose  
    def _get_rank_address(self, id):
        return self._get_prefix_ranks() + \
            _hash(id.encode('utf-8'))[0:64]

    def _get_consistency_address(self):
        return _hash(CONSISTENCY_NS.encode('utf-8'))[0:70]        

    # For each CLI command, add a method to:
    # 1. Do any additional handling, if required
    # 2. Create a transaction and a batch
    # 3. Send to REST API

    def regmethod(self, method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr):
        '''register a method in the ledger.'''
        return self._wrap_and_send("regmethod", method, pcd, lc, crrs, crsr, sc, ad, mdp, mdr, wait=10)
    
    def methods(self):
        addr_prefix = self._get_prefix_methods()

        result = self._send_to_rest_api(
            "state?address={}".format(addr_prefix))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None
    
    def consistency(self, approval):
        '''register the consistency result in the ledger.'''
        return self._wrap_and_send("consistency", approval, wait=10)
    
    def isConsistent(self):
        addr_prefix = self._get_consistency_address()

        result = self._send_to_rest_api(
            "state?address={}".format(addr_prefix))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None

    def reg_data_utility(self, qis):
        '''register data utility QI list in the ledger.'''
        return self._wrap_and_send("data_utility", qis, wait=10)

    def reg_data_privacy(self, qis):
        '''register data privacy QI list in the ledger.'''
        return self._wrap_and_send("data_privacy", qis, wait=10)

    def qis(self):
        addr_prefix = self._get_prefix_qis()

        result = self._send_to_rest_api(
            "state?address={}".format(addr_prefix))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None

    def reg_category_weight(self, purpose, data_utility, data_privacy):
        '''register category weight list for a Purpose in the ledger.'''
        return self._wrap_and_send("category_weight", purpose, data_utility, data_privacy, wait=10)

    def cws(self):
        addr_prefix = self._get_prefix_cws()

        result = self._send_to_rest_api(
            "state?address={}".format(addr_prefix))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None

    def reg_wm_plus(self, purpose, PCD, LC, CRRS, CRSR, SC, AD, MDP, MDR):
        '''register category weight list for a Purpose in the ledger.'''
        return self._wrap_and_send("WM_plus", purpose, PCD, LC, CRRS, CRSR, SC, AD, MDP, MDR, wait=10)

    def wmps(self):
        addr_prefix = self._get_prefix_wmps()

        result = self._send_to_rest_api(
            "state?address={}".format(addr_prefix))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None

    def reg_wm_minus(self, purpose, PCD, LC, CRRS, CRSR, SC, AD, MDP, MDR):
        '''register category weight list for a Purpose in the ledger.'''
        return self._wrap_and_send("WM_minus", purpose, PCD, LC, CRRS, CRSR, SC, AD, MDP, MDR, wait=10)

    def wmms(self):
        addr_prefix = self._get_prefix_wmms()

        result = self._send_to_rest_api(
            "state?address={}".format(addr_prefix))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None

    def delmethod(self, method):
        '''delete a registered method.'''
        return self._wrap_and_send("delmethod", method, wait=10) 

    def compute(self, purpose, m1, m2, m3, m4, m5, m6, m7, m8):
        '''compute QoS of methods for a purpose.'''
        return self._wrap_and_send("compute", purpose, m1, m2, m3, m4, m5, m6, m7, m8, wait=10)

    def rank(self, purpose):
        addr_rank = self._get_rank_address(purpose)

        result = self._send_to_rest_api(
            "state?address={}".format(addr_rank))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None

    def ranks(self):
        addr_rank = self._get_prefix_ranks()

        result = self._send_to_rest_api(
            "state?address={}".format(addr_rank))

        try:
            encoded_entries = yaml.safe_load(result)["data"]

            return [
                base64.b64decode(entry["data"]) for entry in encoded_entries
            ]

        except BaseException:
            return None                           

    def _send_to_rest_api(self, suffix, data=None, content_type=None):
        '''Send a REST command to the Validator via the REST API.

           Called by ranks(), methods(), qis(), wmms(), wmps(), cws() &  _wrap_and_send().
           The latter caller is made on the behalf of regmethod(), reg_data_utility(), reg_data_privacy(), reg_wm_plus(), reg_wm_minus(), and compute().
        '''
        url = "{}/{}".format(self._base_url, suffix)
        print("URL to send to REST API is {}".format(url))

        headers = {}

        if content_type is not None:
            headers['Content-Type'] = content_type

        try:
            if data is not None:
                result = requests.post(url, headers=headers, data=data)
            else:
                result = requests.get(url, headers=headers)

            if not result.ok:
                raise Exception("Error {}: {}".format(
                    result.status_code, result.reason))
        except requests.ConnectionError as err:
            raise Exception(
                'Failed to connect to {}: {}'.format(url, str(err)))
        except BaseException as err:
            raise Exception(err)

        return result.text

    def _wait_for_status(self, batch_id, wait, result):
        '''Wait until transaction status is not PENDING (COMMITTED or error).

           'wait' is time to wait for status, in seconds.
        '''
        if wait and wait > 0:
            waited = 0
            start_time = time.time()
            while waited < wait:
                result = self._send_to_rest_api("batch_statuses?id={}&wait={}"
                                               .format(batch_id, wait))
                status = yaml.safe_load(result)['data'][0]['status']
                waited = time.time() - start_time

                if status != 'PENDING':
                    return result
            return "Transaction timed out after waiting {} seconds." \
               .format(wait)
        else:
            return result


    def _wrap_and_send(self, action, *params, wait=None):
        '''Create a transaction, then wrap it in a batch.

           Even single transactions must be wrapped into a batch.
           Called by regmethod(), reg_data_utility(), reg_data_privacy(), reg_wm_plus(), reg_wm_minus(), and compute() 
        '''
        # Generate a CSV UTF-8 encoded string as the payload.
        if action == "regmethod":
            raw_payload = ",".join([action, params[0], str(params[1]), str(params[2]), str(params[3]), str(params[4]), str(params[5]), str(params[6]), str(params[7]), str(params[8])])
            address_input = [self._get_method_address(params[0])] 
            address_output = address_input
        elif action == "consistency":
            raw_payload = ",".join([action, str(params[0])])
            address_input = [self._get_consistency_address()] 
            address_output = address_input
        elif action == "data_utility":    
            raw_payload = ",".join([action, str(params[0])])
            address_input = [self._get_data_utility_address()]
            address_output = address_input
        elif action == "data_privacy":    
            raw_payload = ",".join([action, str(params[0])])
            address_input = [self._get_data_privacy_address()]
            address_output = address_input
        elif action == "category_weight":    
            raw_payload = ",".join([action, str(params[0]), str(params[1]), str(params[2])])
            address_input = [self._get_category_weight_address(params[0])]
            address_output = address_input
        elif action == "WM_plus":    
            raw_payload = ",".join([action, str(params[0]), str(params[1]), str(params[2]), str(params[3]), str(params[4]), str(params[5]), str(params[6]), str(params[7]), str(params[8])])
            address_input = [self._get_wm_plus_address(params[0])]
            address_output = address_input
        elif action == "WM_minus":    
            raw_payload = ",".join([action, str(params[0]), str(params[1]), str(params[2]), str(params[3]), str(params[4]), str(params[5]), str(params[6]), str(params[7]), str(params[8])])
            address_input = [self._get_wm_minus_address(params[0])]
            address_output = address_input
        elif action == "delmethod":    
            raw_payload = ",".join([action, params[0]])
            address_input = [self._get_method_address(params[0])]
            address_output = address_input
        elif action == "compute":    
            raw_payload = ",".join([action, params[0], str(params[1]), str(params[2]), str(params[3]), str(params[4]), str(params[5]), str(params[6]), str(params[7]), str(params[8])])
            address_input = [self._get_category_weight_address(params[0]), self._get_wm_plus_address(params[0]), self._get_wm_minus_address(params[0]), \
                self._get_method_address(params[1]), self._get_method_address(params[2]), self._get_method_address(params[3]), \
                self._get_method_address(params[4]), self._get_method_address(params[5]), self._get_method_address(params[6]), \
                self._get_method_address(params[7]), self._get_method_address(params[8]),
                self._get_data_utility_address(), self._get_data_privacy_address()]
            address_output = [self._get_rank_address(params[0])]        
                       
        payload = raw_payload.encode() # Convert Unicode to bytes

        # Construct the address where we'll store our state.
        # We just have one input and output address (the same one).        
        #addresses_input = address_input
        # Create a TransactionHeader.
        header = TransactionHeader(
            signer_public_key=self._public_key,
            family_name=FAMILY_NAME,
            family_version="1.0",
            inputs=address_input,
            outputs=address_output,
            dependencies=[],
            payload_sha512=_hash(payload),
            batcher_public_key=self._public_key,
            nonce=random.random().hex().encode()
        ).SerializeToString()

        # Create a Transaction from the header and payload above.
        transaction = Transaction(
            header=header,
            payload=payload,
            header_signature=self._signer.sign(header)
        )

        transaction_list = [transaction]

        # Create a BatchHeader from transaction_list above.
        header = BatchHeader(
            signer_public_key=self._public_key,
            transaction_ids=[txn.header_signature for txn in transaction_list]
        ).SerializeToString()

        # Create Batch using the BatchHeader and transaction_list above.
        batch = Batch(
            header=header,
            transactions=transaction_list,
            header_signature=self._signer.sign(header))

        # Create a Batch List from Batch above
        batch_list = BatchList(batches=[batch])
        batch_id = batch_list.batches[0].header_signature

        # Send batch_list to the REST API
        result = self._send_to_rest_api("batches",
                                       batch_list.SerializeToString(),
                                       'application/octet-stream')

        # Wait until transaction status is COMMITTED, error, or timed out
        return self._wait_for_status(batch_id, wait, result)


# Synthetic data generation method ranking
Sawtooth Blockchain implementation of the synthetic data generation methods ranking

# Prerequisites for running the smart contract and the client:
## On the Sawtooth node:
* sudo apt-get update 
* sudo apt-get install curl -y 
* sudo curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x8AA7AF1F1091A5FD' | sudo apt-key add - 
* sudo chmod u+x /etc/apt/sources.list 
* sudo echo "deb [arch=amd64] http://repo.sawtooth.me/ubuntu/bumper/stable xenial universe" >> /etc/apt/sources.list 
* sudo apt-get update 
* sudo apt-get install -y -q apt-transport-https build-essential ca-certificates 
* sudo apt-get clean 
* sudo rm -rf /var/lib/apt/lists/* 
## On the client:
* apt-get update;  apt-get install curl -y;  echo "deb http://repo.sawtooth.me/ubuntu/ci xenial universe" >> /etc/apt/sources.list  && curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x8AA7AF1F1091A5FD' | apt-key add - 
* echo "deb [arch=amd64] http://repo.sawtooth.me/ubuntu/bumper/stable xenial universe" >> /etc/apt/sources.list 
* apt-get update  && apt-get install -y -q --no-install-recommends     apt-utils  && apt-get install -y -q     apt-transport-https     build-essential     ca-certificates     inetutils-ping     libffi-dev     libssl-dev     python3-aiodns=1.1.1-1     python3-aiohttp>=2.3.2-1     python3-aiopg     python3-async-timeout=1.2.0-1     python3-bitcoin=1.1.42-1     python3-cbor     python3-cchardet=2.0a3-1     python3-chardet=2.3.0-1     python3-colorlog     python3-cov-core     python3-cryptography-vectors=1.7.2-1     python3-cryptography=1.7.2-1     python3-dev     python3-grpcio-tools=1.1.3-1     python3-grpcio=1.1.3-1     python3-lmdb=0.92-1     python3-multidict=2.1.4-1     python3-netifaces=0.10.4-0.1build2     python3-nose2     python3-pip     python3-protobuf     python3-psycopg2     python3-pycares=2.1.1-1     python3-pyformance     python3-pytest-runner=2.6.2-1     python3-pytest=2.9.0-1     python3-pytz=2016.10-1     python3-requests     python3-secp256k1=0.13.2-1     python3-setuptools-scm=1.15.0-1     python3-six=1.10.0-1     python3-toml     python3-yaml     python3-yarl=0.10.0-1     python3-zmq     software-properties-common     python3-sawtooth-sdk     python3-sawtooth-cli  && apt-get clean  && rm -rf /var/lib/apt/lists/* 

# How to run the Smart Contract on Sawtooth nodes:
We have to run the smart contract on all the Sawtooth nodes contributing to the consensus by the following command: sudo python3 ./synthrank_tp-v4.py

![image](https://github.com/mhtaba/synthrank/assets/111292110/8908caf2-406d-4e5f-ad0d-5ee2136b56ce)

# How to run the client:
sudo python3 ./synthrank.py {one_of_the_following_commands}
1. Registering quality indicators written in a file by the product manager:
   * qi qi.txt --key {user}
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/30e0acc5-dea4-4953-8674-cf349ad38dc5)
2. Registering category weights written in a file by the product manager:
   * cw weights.txt --key {user}
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/0f287ae4-8049-4160-a166-2ceac3b51f7f)
3. Registering WM_plus written in a file by the product manager:
   * wmp WM_plus.txt --key {user}
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/3e51dd08-7124-4a30-b5ee-8161054e4303)
4. Registering WM_minus written in a file by the product manager:
   * wmm WM_minus.txt --key {user}
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/3bda4991-7b29-4c55-8863-0b926853f16d)
5. Registering methods written in a file by the data scientist:
   * method inputs.txt --key {user}
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/7f255bbc-fd80-4866-a21d-a9322cdaf283)
6. Computing QoS score of all the registered methods for a given purpose written in a file and registering the ranked result:
   * qos compute.txt
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/7d796981-c32a-49b4-8d97-f185cd5512b6)
7. Getting the QoS score rank of all the registered methods for a given purpose:
   * rank {scenario_name}
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/0ebe0b5f-2416-4d32-a35b-a524bd63ab40)
8. Getting the QoS score rank of all the registered purposes:
   * ranks
    ![image](https://github.com/mhtaba/synthrank/assets/111292110/4565537c-3e17-4630-b23c-de58858409e2)
9. Auditing the files given to the auditor and comparing with the registered data in the ledger by the auditor, showing the audit process, and registering the final audit result:
   * audit --key {user}
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/6ec93b77-25d3-4768-a4a1-634378ce3c9f)
10. Getting the registered audit result:
    * isConsistent
      ![image](https://github.com/mhtaba/synthrank/assets/111292110/99612269-a5e9-4927-a27d-5b3155c87421)
11. Getting the registered methods:
    * methods
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/f6e58c5d-9b09-4751-8a70-9e02673a71ac)
12. Getting the registered category weights:
    * cws
      ![image](https://github.com/mhtaba/synthrank/assets/111292110/11dd7747-693f-4151-8f79-e294b986c49e)
13. Getting the registered wm_plus:
    * wmps
      ![image](https://github.com/mhtaba/synthrank/assets/111292110/b2b0e9c8-511b-42eb-98b6-9e3ae2d76fa0)
14. Getting the registered wm_minus:
    * wmms
      ![image](https://github.com/mhtaba/synthrank/assets/111292110/18e55193-dbc8-4dd9-8601-3a130702630b)
15. Getting the registered quality indicators:
    * qis
      ![image](https://github.com/mhtaba/synthrank/assets/111292110/71ef8a77-e5f6-463e-8801-d32b83df3b98)
    
## Format of the given files to the application has to be as follows:
1. inputs.txt for giving the methods and their values filled like this:
   * regmethod im --pcd 0.9 --lc -3.62 --crrs 0.94 --crsr 1.0 --sc 1.0 --ad 0.325 --mdp 0.497 --mdr 0.97
   * input.txt file Example:
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/d8cbea57-e9d5-41a3-9315-7d08f1671c06)
2. qi.txt for giving the quality metrics filled like this:
   * data_utility PCD CRRS CRSR LC SC
   * data_privacy AD MDP MDR
   * qi.txt file example:
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/6595864f-d6e7-4fef-96cb-e69fa700b7de)
3. weights.txt for category weights of a purpose filled like this:
   * category_weight PurposeA --data_utility 0.5 --data_privacy 0.5
   * weights.txt file example:
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/89abcecc-9a81-46cd-81ed-204a241a7757)
4. WM_plus.txt for WM plus of a purpose filled like this:
   * WM_plus PurposeA --PCD 0.20 --CRRS 0.20 --CRSR 0.20 --AD 0.20
   * WM_plus.txt file example:
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/62d67cc2-6bcd-408d-89fd-cf777ab63d7f)
5. WM_minus.txt for WM minus of a purpose filled like this:
   * WM_minus PurposeA --LC 0.25 --SC 0.25 --MDR 0.25 --MDP 0.25
   * WM_minus.txt file example:
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/773ccd1b-8f1a-4911-9cf0-0d39a999b1b0)
6. compute.txt for computing the QoS of a puprpose with given methods filled like this:
   * compute PurposeA im bn mpom clgp mc_medgan mice_lr mice_lr_desc mice_dt
   * compute.txt file example:
     ![image](https://github.com/mhtaba/synthrank/assets/111292110/9925c9e0-94d5-4e25-a87c-a662baaf5eeb)

All of the files' contents have to be finished with {end}. 
 

import logging

password = 'password'
user = 'user'
db = 'zsa'
port = '5432'
host = 'host'
target = open('/path/to/target.sdf', encoding='UTF-8')
steps = 3000
popsize = 50
chromo_len = 10
pop_dict_name = 'populations.pickle'

logger = logging.getLogger("GA")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("/path/to/ga.log", mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

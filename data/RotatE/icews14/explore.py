import os

def load_dict(filename):
    with open(filename, encoding="utf8") as fin:
        item2id = dict()
        i = 0
        for line in fin:
            # previous code
            # key, value = line.strip().split('\t')
            # new code
            line = line.strip()
            middle = len(line) // 2
            key, value = line[:middle], line[middle:]
            key, value = key.strip(), value.strip()
            try:
                item2id[value] = int(key)
                i = int(key) + 1
            except ValueError:
                item2id[key] = i
                i += 1
    return item2id

def read_triple(file_path, entity2id=None, relation2id=None):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            h, r, t = h.strip(), r.strip(), t.strip()
            try: 
                triples.append((int(h) if entity2id is None else entity2id[h],
                            int(r) if relation2id is None else relation2id[r],
                            int(t) if entity2id is None else entity2id[t]))
            except ValueError as e:
                print("ERROR: ", line, e)
    return triples

file_names = list(filter(lambda x: not x.endswith('.py'), os.listdir('.')))
entity2id = load_dict('entities.dict')
relation2id = load_dict('relations.dict')
print('[entity2id]', len(entity2id))
print('[relation2id]', len(relation2id))

train_triples = read_triple('train.txt', entity2id=entity2id, relation2id=relation2id)
valid_triples = read_triple('valid.txt', entity2id=entity2id, relation2id=relation2id)
test_triples = read_triple('test.txt', entity2id=entity2id, relation2id=relation2id)
print('[train_triples]', train_triples[:10])
print('[valid_triples]', valid_triples[:10])
print('[test_triples]', test_triples[:10])

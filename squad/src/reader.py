import json

with open('../data/dev-v1.1.json') as f:
    for line in f:
        parsed = json.loads(line)['data'][0]
        print json.dumps(parsed)
        break
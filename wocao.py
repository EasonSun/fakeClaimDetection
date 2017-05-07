from urllib.parse import urlparse
import json
import os
import io

googleDataPath="data/Googlee"
for filePath in os.listdir(googleDataPath):
	if not filePath.endswith('.json'):
		pass
	filePath = os.path.join(googleDataPath, filePath)
	newData = {'article': [], 'source':[]}
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	for item in data:
		newData['article'].append(item['html'])
		newData['source'].append(urlparse(item['url']).netloc)
	os.remove(filePath)
	with open(filePath, 'w') as f: 
		json.dump(newData, f, indent=4)
	break
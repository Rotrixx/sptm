import argparse

parser = argparse.ArgumentParser(description='sptm multilabel')
parser.add_argument("-f1", help="file 1")
parser.add_argument("-f2", help="file 2")
args = parser.parse_args()

multilabelFile = []

with open(args.f1,'r') as file:
	for line in file:
		multilabelFile.append(line)

i = 0
with open(args.f2,'r') as file:
	for line in file:
		if line == 'subtask_a':
			pass
		if line != multilabelFile[i]:
			line = line.strip('\n')
			line = line.split('\t')
			multilabelFile[i] = multilabelFile[i].strip('\n') + '\t' + line[1] + '\n'
		i += 1

with open('OutputData/multilabelFinalRun.txt','w') as file:
	for i in multilabelFile:
		file.write(i)
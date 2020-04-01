import sys

f1=open(sys.argv[1],"r")
f2=open(sys.argv[2],"r")

base=[]
comp=[]
for line in f1:
    try:
        base.append(int(line))
    except ValueError:
        print()

for line in f2:
    try:
        comp.append(int(line))
    except:
        print()

correct = 0
for i in range(0,len(base)):
    if comp[i] == base[i]:
        correct = correct + 1

print(correct/len(base)*100)

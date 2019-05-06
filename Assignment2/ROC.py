#firas-308185313
#shirin- 311382840
import matplotlib.pyplot as plt
colors = ['black', 'red', 'darkorange', 'cornflowerblue', 'yellow','pink']
Arectol=[0.9,0.8,0.7,0.9]
Erectol=[0.9,0.9,0.9,0.9]
Brectol=[1.,0.9,1.,0.9]
Apretol=[0.9,0.888889,0.875000,0.818182]
Epretol=[1.,0.818182,0.9,1.]
Bpretol=[0.909091,0.9,0.833333,0.9]
Arec=[0.6,0.6,0.9,0.7,0.9]
Erec=[0.5,0.7,0.7,0.7,0.8]
Brec=[0.6,0.6,0.9,0.8]
Apre=[0.461538,0.666667,0.692308,0.636364,0.750000]
Epre=[0.625,0.583333,1.,0.777778,0.888889]
Bpre=[0.666667,0.666667,0.9,0.8]
clas= ['Airplane changing tol in Kmeans','Bike changing tol in Kmeans','Elephant changing tol in Kmeans','Airplane changing # clusters','Elephant changing # clusters','Bike changing # clusters']
recalls=[Arectol,Erectol,Brectol,Arec,Erec,Brec]
precisions=[Apretol,Epretol,Bpretol,Apre,Epre,Bpre]
plt.figure(figsize=(5, 7))
lines = []
labels = []
for i, color in zip(range(6), colors):
    l, = plt.plot(recalls[i], precisions[i], color=color, lw=2)
    lines.append(l)
    labels.append('{0}'
                  ''.format(clas[i]))
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ROC for all classes using different methods and parameters')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.show()
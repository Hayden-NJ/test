def clfscore(clf, x, y, pos_label=1):
    predict = clf.predict(x)
    name = ['precision','recall','f1','accuracy']
    command = ['metrics.precision_score(y, predict)',
              'metrics.recall_score(y, predict)',
              'metrics.f1_score(y, predict)',
              'metrics.accuracy_score(y, predict)',
              ]
    for i in range(4):
        print(name[i], eval(command[i]))
    fpr, tpr, thresholds = metrics.roc_curve(y, predict, pos_label=pos_label)
    print('AUC: ', metrics.auc(fpr, tpr))
    
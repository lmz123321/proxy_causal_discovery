def precision(preds,targs):
    '''
    pre = true_pred / all_preds
    '''
    if len(preds)==0:
        return None
    tp = 0
    for pred in preds:
        if pred in targs:
            tp += 1
    return tp/len(preds)

def recall(preds,targs):
    '''
    recall = true_pred / all_targets
    '''
    if len(targs)==0:
        return None
    tp = 0
    for pred in preds:
        if pred in targs:
            tp += 1
    return tp/len(targs)

def f1(preds,targs):
    '''
    F1 score := 2(pr*re)/(pr+re)
    '''
    pr = precision(preds,targs)
    re = recall(preds,targs)
    if pr is None or re is None:
        return None
    elif (pr+re)==0:
        return 0
    else:
        f1 = 2*pr*re/(pr+re)
        return f1
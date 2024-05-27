def precision_skeleton(pred,gt):
    '''
    Precision of the detected skeleton
    Args: pred, a list of predicted *undirected* edges
          gt, ground-truth *undirected* egde list
    '''
    num_fp = 0
    for edge in pred:
        begin,end = edge
        if edge not in gt and (end,begin) not in gt:
            num_fp +=1
    return 1 - num_fp/len(pred)


def recall_skeleton(pred,gt):
    '''
    Recall of the detected skeleton
    Args: pred, a list of *predicted* undirected edges
          gt, ground-truth *undirected* egde list
    '''
    num_fn = 0
    for edge in gt:
        begin,end = edge
        if edge not in pred and (end,begin) not in pred:
            num_fn +=1
    return 1 - num_fn/len(gt)


def precision(pred,gt):
    '''
    Precision of the detected sDAG
    Agrs: pred, a list of predicted *directed* edges
          gt, ground-truth *directed* edges
    '''
    num_fp = 0
    for edge in pred:
        if edge not in gt:
            num_fp += 1
    return 1-num_fp/len(pred)

def recall(pred,gt):
    '''
    Recall of the detected sDAG
    Agrs: pred, a list of predicted *directed* edges
          gt, ground-truth *directed* edges
    '''
    num_fn = 0
    for edge in gt:
        if edge not in pred:
            num_fn += 1
    return 1-num_fn/len(gt)
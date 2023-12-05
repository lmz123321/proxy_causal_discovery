def norm(data):
    '''
    mean/std normalization
    '''
    return (data-data.mean())/data.std()
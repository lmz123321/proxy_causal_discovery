import numpy as np

def merge(infos):
    '''
    merge in list in each infos[key]
    '''
    
    _info = infos[0]; keys = _info.keys()
    info = dict()
    for key in keys:
        info[key] = list()

    for _info in infos:
        for key in keys:
            info[key].append(_info[key])

    for key in keys:
        info[key] = np.stack(info[key])
        
    return info

def remove_none(arr,verbose=True):
    '''
    remove None in a numpy.array
    for 2d array, remove any row with None; for 1d array, remove any element with None
    '''
    if len(arr.shape)==2:
        nonerows = list(set(np.where(arr==None)[0]))
        if verbose:
            print('{} rows with None.'.format(len(nonerows)))
        return np.delete(arr,nonerows,axis=0).astype(np.float)
    elif len(arr.shape)==1:
        return arr[arr!=None].astype(np.float)
    else:
        raise ValueError('only take 1d or 2d array, got {}d.'.format(len(arr.shape)))
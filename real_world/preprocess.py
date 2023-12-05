import os
import scipy.io
import pandas as pd
import numpy as np

def delete_digit(string):
    """
    Functs: - delete digits at the begining and the ending of AAL biomarker names
            - e.g. string='108 Cerebelum_10_R 9082\n'
            - return 'Cerebelum_10_R'
    """
    out = []
    llen = len(string)
    for ind in range(llen):
        i = string[ind]
        if ind <= 3 and i.isdigit():
            out.append(i)

    return 'X'+''.join(out)


def replace_str(st):
    r"""
    Functs: - replce the - and / in the input str into ''
    """
    rp = st.replace('-', ' ').replace('/', '')

    return rp

def map_diag_labels(value):
    r"""
    Runcts: - take a value as input, which may be NC,SMC,EMCI,LMCI,AD,NaN
            - aggregate EMCI and LMCI into MCI
            - other classes remain unchanged
    """
    if 'MCI' in value:
        return 'MCI'
    else:
        return value


def list_add(aa, bb):
    r"""
    Functs: - element wise list adding
    """
    assert not (len(aa) == 0 and len(bb) == 0)
    if len(aa) == 0:
        aa = [0 for b in bb]
    if len(bb) == 0:
        bb = [0 for a in aa]

    ssum = [sum(x) for x in zip(aa, bb)]

    return ssum


vis_reformat = {'ADNI-Screening':'ADNI1_B',
                'ADNIGO-Screening-MRI':'ADNI1_B','ADNI-Baseline':'ADNI1_B',
              'ADNIGO-Month-3-MRI':'ADNI1_M3', 'ADNI1/GO-Month-6':'ADNI1_M6',
              'ADNI1/GO-Month-12':'ADNI1_M12','ADNI1/GO-Month-18':'ADNI1_M18',
              'ADNI1/GO-Month-24':'ADNI1_M24','ADNI1/GO-Month-36':'ADNI1_M36',
              'ADNI1/GO-Month-48':'ADNI1_M48','ADNIGO-Month-60':'ADNI1_M60',
              'ADNI2-Screening-MRI-New-Pt':'ADNI2_B','ADNI2-Initial-Visit-Cont-Pt':'ADNI2_B',
              'ADNI2-Month-3-MRI-New-Pt':'ADNI2_M3','ADNI2-Month-6-New-Pt':'ADNI2_M6',
              'ADNI2-Year-1-Visit':'ADNI2_M12','ADNI2-Year-2-Visit':'ADNI2_M24',
              'ADNI2-Year-3-Visit':'ADNI2_M36','ADNI2-Year-4-Visit':'ADNI2_M48',
              'ADNI2-Year-5-Visit':'ADNI2_M60',
              'ADNI3-Initial-Visit-Cont-Pt':'ADNI3_B',
              'ADNI3-Year-1-Visit':'ADNI3_M12','ADNI3-Year-2-Visit':'ADNI3_M24',
              'No-Visit-Defined':None,'ADNI2-Tau-only-visit':None}


class ADNILongData():
    def __init__(self, ):
        self.base = '/data/liumingzhou/ADNI/Longitudinal_30T/'
        self.idfilename = 'AAL_Fns.mat'
        self.aalfilename = 'AAL_feature_all.mat'
        self.gmfilename = '_GM.mat'
        self.wmfilename = '_WM.mat'
        self.diagblfilename = 'DXSUM_bl.mat'
        self.aggrdata = {'subject': [], 'visit': []}

        filename = '/data/liumingzhou/ADNI/aal.txt'
        with open(filename, 'r') as f_in:
            lines = f_in.readlines()
            f_in.close()
        self.aal_indices = [delete_digit(biom) for biom in lines if biom != '\n']

        self.read()

    def read(self, ):
        '''
        Read id information (data_id, visit_id) and AAL features (90 brain regions); load them into self.aggrData
        Visit naming rules:
           ADNI-1 including('ADNI','ADNI GO','ADNI 1/GO')
           - start with a 'ADNIGO-Screening-MRI' visit or a 'ADNI-Baseline' visit.
           - follow up with 'ADNIGO-Month-3', 'ADNI1/GO-Month-6,12,18,24,36,48'
           ADNI-2
           - start with a 'ADNI2-Screening-MRI-New-Pt' or a 'ADNI2-Initial-Visit-Cont-Pt' visit
           - follow up with 'ADNI2-Month-3-MRI-New-Pt', 'ADNI2-Month 6-New-Pt', 'ADNI2-Year-1,2,3,4,5-Visit'
           ADNI-3
           - start with a 'ADNI3-Initial-Visit-Cont-Pt' vist
           - follow up with 'ADNI3-Year 1,2-Visit'

        A major difference between ADNILongData and ADNIData is that we need to normalize brain volumes into TIV features ourself
        - specifically, divide the orig volume with corresponding volume of grey matter plus white matter
        - we have confirmed (in 5_check_mat_content_30T_long.ipynb) the filename order is the same between features_all.mat and _GM/_WM.mat
        '''
        # subject_id and visit_id
        orig_idinfos = scipy.io.loadmat(os.path.join(self.base, self.idfilename))['Fns']

        idinfos = list()
        for ind in range(orig_idinfos.shape[0]):
            idinfos.append(orig_idinfos[ind, 0][0])

        for idinfo in idinfos:
            names = idinfo.split('\\')[-1]
            self.aggrdata['subject'].append('_'.join([names.split('_')[0][-3:]] + names.split('_')[1:3]))  # 002,S,0295
            orig_visit = '/'.join(names.split('_')[3:-1])
            self.aggrdata['visit'].append(vis_reformat[orig_visit])

        # label
        '''
        filename = os.path.join(self.base, self.diagblfilename)
        origdiags = scipy.io.loadmat(filename)['DXSUM']
        diags = [d[0] if d.shape == (1,) else 'nan' for d in origdiags[:, 0]]
        self.aggrdata['label'] = list(map(map_diag_labels, diags))
        '''

        # brain volumns
        origvols = scipy.io.loadmat(os.path.join(self.base, self.aalfilename))['feature_all']
        gms = scipy.io.loadmat(os.path.join(self.base, self.gmfilename))['results']
        wms = scipy.io.loadmat(os.path.join(self.base, self.wmfilename))['results']
        aaldata = origvols / (gms[0, 1].T + wms[0, 1].T)

        biomarker_vols = [list(d) for d in aaldata.T]
        bioData = dict(zip(self.aal_indices, biomarker_vols))

        for aal_index in self.aal_indices:
            self.aggrdata[aal_index] = bioData[aal_index]

        self.aggrdf = pd.DataFrame(self.aggrdata)

    def select(self, num_nodes, begin, finish):
        '''
        select subjects with two consecutive follow-ups, for subjects enrolled twice, he/she
        will be treated as two independent subjects (to enlarge sample size)
        '''
        sub_to_vis = dict()
        sub_to_label = dict()
        aggr = self.aggrdata
        for ind, subject in enumerate(aggr['subject']):
            visit = aggr['visit'][ind]
            if visit is not None:
                if subject not in sub_to_vis.keys():
                    sub_to_vis[subject] = list()
                sub_to_vis[subject].append(visit)
            '''
            label = aggr['label'][ind]
            if label!='nan':
                if subject not in sub_to_label.keys():
                    sub_to_label[subject] = list()
                sub_to_label[subject].append(label!='CN')
            '''

        # select
        selected = list()
        num = 0
        for trial in ['ADNI1', 'ADNI2', 'ADNI3']:
            for subject in sub_to_vis.keys():
                start = '{}_{}'.format(trial, begin)
                end = '{}_{}'.format(trial, finish)

                # adflag = True if the subject is diagnozed as MSC/MCI/AD
                '''
                adflag = False
                if subject in sub_to_label.keys():
                    if np.array(sub_to_label[subject]).any():
                        adflag = True
                '''
                if start in sub_to_vis[subject] and end in sub_to_vis[subject]:
                    num += 1
                    selected.append((subject, start, end))
        print('{}-->{}: has {} subjects'.format(begin, finish, num))

        # prepare selected data
        output = list()
        df = self.aggrdf
        for subject, start, end in selected:
            # (90,)
            start_array = df[(df['subject'] == subject) & (df['visit'] == start)].iloc[-1, 2:num_nodes + 2].values
            # (90,)
            end_array = df[(df['subject'] == subject) & (df['visit'] == end)].iloc[-1, 2:num_nodes + 2].values
            # (180,)
            array = start_array.tolist() + end_array.tolist()
            output.append(array)

        # num_samples, 180(X1_0,...,X90_0,X1_1,...,X90_1)
        output = np.array(output)
        return output

if __name__ == '__main__':
    begin = 'M6'
    finish = 'M12'
    num_nodes = 20

    data = ADNILongData()
    output = data.select(num_nodes, begin, finish)
    np.save('./data/ADNI30T3729_{}to{}_d{}'.format(begin, finish, num_nodes), output)

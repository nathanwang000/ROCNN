'''
In this file I would like to parse the output
specifically, I want to make a summary of all the files in the output folder (excluding output of non models)

I care about
1) the model name (done)
2) model specification: lr, wd, rp (done)
3) losses: a vector of losses (done)
4) accuracy: on validation set (done)
5) confusion matrices (done)
'''
import os
def parseOutput(filename):
    ''' 
    return (bool, dic) where bool indicate file is outputfile
    and dic is of the shape {LearningRate:0.1,...}
    '''
    res = {}
    with open(filename) as f:
        # read the first 7 lines
        index = 1
        switch = {
            1: 'Model',
            2: 'LearningRate',
            3: 'Momentum',
            4: 'LearningRateDecay',
            5: 'WeightDecay',
            6: 'BatchSize',
            7: 'RotationPrior'
        }
        convert = [str,float,float,float,float,int,float]
        #convert = [str for i in range(7)]
        lines = f.readlines()
        for l in lines:
            if l.find(switch[index])==-1: return False, res
            res[switch[index]] = convert[index-1](l[l.find(':')+2:l.rfind('\x1b[0m')])
            index = index + 1
            if index==7: break
        res['loss'] = {'tr': [], 'te': []}
        res['confusion'] = {'tr': [], 'te': []}
        index = 0
        while index < len(lines):
            l = lines[index]
            # deprecated: uncomment the following lines when ready
            if l.find('loss') != -1: res['loss']['tr'].append(float(l[l.find('was')+4:l.rfind('\x1b[0m')]))
            '''
            if l.find('tr loss') != -1: res['loss']['tr'].append(float(l[l.find('was')+4:l.rfind('\x1b[0m')]))
            if l.find('te loss') != -1: res['loss']['te'].append(float(l[l.find('was')+4:l.rfind('\x1b[0m')]))
            '''
            # read in final confusion matrix for both train and test
            if l.find('time to learn 1 sample') != -1:
                l = ""
                index = index + 2
                while index < len(lines) and lines[index].find('global correct') == -1:
                    l = l + lines[index]
                    index = index + 1
                if index < len(lines): res['confusion']['tr'].append(l+lines[index]) 
            if l.find('time to test 1 sample') != -1:
                l = ""
                index = index + 2
                while index < len(lines) and lines[index].find('global correct') == -1:
                    l = l + lines[index]
                    index = index + 1
                if index < len(lines): res['confusion']['te'].append(l+lines[index]) 

            index = index + 1

    # read train.log and test.log to get information about accuracy
    '''' 
    try just use the confusion matrix learned
    '''
    # res['acc'] = {'tr': [], 'te': []}
    # with open(os.path.join(res['Model'],'train.log')) as f:
    #     for l in f:
    #         try: res['acc']['tr'].append(float(l))
    #         except: continue
    # with open(os.path.join(res['Model'],'test.log')) as f:
    #     for l in f:
    #         try: res['acc']['te'].append(float(l))
    #         except: continue
        
    return True, res

def parseAll(directory):
    ''' 
    parseOutput for every file in directory
    '''
    files = os.listdir(directory)
    outputs = []
    for f in files:
        valid, output = parseOutput(os.path.join(directory,f))
        if valid: outputs.append(output)
    return outputs

if __name__ == '__main__':
    outputs = parseAll('outputs/')

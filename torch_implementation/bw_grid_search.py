import os

modelName = "lenetF"
pbs_file = '''
#!/bin/bash
#PBS -N %s
#PBS -M jiaxuan@umich.edu
#PBS -m abe

#PBS -V
#PBS -j oe

#PBS -l nodes=5:xe
#PBS -l mem=20g
#PBS -l walltime=10:00:00

####  End PBS preamble
#  Put your job commands after this line
cd ''' + os.path.dirname(os.path.realpath(__file__)) + '''

th trainOnMnistRot.lua %s -model ''' + modelName + '''
echo "process complete"
'''

def torchModel(modelName=modelName,newLoss=False):
    saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    command = "--val --full --save %s -r %f --coefL2 %f --rp %f -b %d -m %f -t 5"
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
    lrs = [1e-1,1e-2,1e-3,1e-4];
    batchSizes = [100,400];
    rotPriors = [5e-1,5e-2,5e-3,5e-4];
    wds = [5e-1,5e-2,5e-3];
    mus = [0];
    # archetectural changes: manually set
    # filterSizes = [5,10,15,20,25];
    # nFilters = [20,50,80];
    # poolSizes = [2,3,4];
    index = 1
    for lr in lrs:
        for bs in batchSizes:
            for rp in rotPriors:
                for wd in wds:
                    for mu in mus:
                        name = '%s%d' % (modelName,index)
                        with open(name+'.pbs', 'w') as f:
                            f.write(pbs_file % (name,command%(saveFN%index,lr,wd,rp,bs,mu)))
                        index = index + 1


def torchModelNoRP(modelName=modelName,newLoss=False):
    saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    command = '--val --full --save %s -r %f --coefL2 %f --rp %f -b %d -m %f -t 5'
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
    lrs = [1e-1,1e-2,1e-3,1e-4];
    batchSizes = [100,400];
    rotPriors = [0];
    wds = [5e-1,5e-2,5e-3];
    mus = [0,0.1];
    # archetectural changes: manually set
    # filterSizes = [5,10,15,20,25];
    # nFilters = [20,50,80];
    # poolSizes = [2,3,4];
    index = 97
    for lr in lrs:
        for bs in batchSizes:
            for rp in rotPriors:
                for wd in wds:
                    for mu in mus:
                        name = '%s%d' % (modelName,index)
                        with open(name+'.pbs', 'w') as f:
                            f.write(pbs_file % (name,command%(saveFN%index,lr,wd,rp,bs,mu)))
                        index = index + 1

def torchModelLrd(modelName=modelName,newLoss=False):
    saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    command = '--rpLastLayer true --val --full --save %s -r %f --coefL2 %f --rp %f -b %d -t 5 -lrd %f -m %f'
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
    lrs = [1e-1,1e-2];
    lrds = [5e-4,5e-3,5e-2]
    bs = 400
    rotPriors = [5e-1,5e-2,5e-3,5e-4];
    wds = [5e-1,5e-2,5e-3];
    mus = [0];
    # archetectural changes: manually set
    # filterSizes = [5,10,15,20,25];
    # nFilters = [20,50,80];
    # poolSizes = [2,3,4];
    index = 145
    for lr in lrs:
        for lrd in lrds:
            for rp in rotPriors:
                for wd in wds:
                    for mu in mus:
                        name = '%s%d' % (modelName,index)
                        with open(name+'.pbs', 'w') as f:
                            f.write(pbs_file % (name,command%(saveFN%index,lr,wd,rp,bs,lrd,mu)))
                        index = index + 1

def DSNSetting(modelName=modelName,newLoss=False):
    saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    command = '--val --save %s -r %f --coefL2 %f --rp %f -b %d -t 5 -lrd %f --trsize %d --tesize %d'
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
    index = 217 # index starts at 217
    notches = [100,200,400,800,2500,4000,7000]
    lr = 0.01; wd = 0.005; rp = 0; bs = 100; lrd = 5e-5; # based on the best model setting
    for n in notches:
        name = '%s%d' % (modelName,n)
        with open(name+'.pbs', 'w') as f:
            f.write(pbs_file % (name,command%(saveFN%index,lr,wd,rp,bs,lrd,n,2000)))
        index = index + 1

def torchModelNoWD(modelName=modelName,newLoss=False):
    saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    command = "--rpLastLayer --val --full --save %s -r %f --coefL2 %f --rp %f -b %d -m %f -t 5"
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
        saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    lrs = [1e-1,1e-2,1e-3,1e-4];
    batchSizes = [100,400];
    rotPriors = [5e-1,5e-2,5e-3,5e-4];
    wds = [0];
    mus = [0];
    # archetectural changes: manually set
    # filterSizes = [5,10,15,20,25];
    # nFilters = [20,50,80];
    # poolSizes = [2,3,4];
    index = 224 # to 255
    for lr in lrs:
        for bs in batchSizes:
            for rp in rotPriors:
                for wd in wds:
                    for mu in mus:
                        name = '%s%d' % (modelName,index)
                        with open(name+'.pbs', 'w') as f:
                            f.write(pbs_file % (name,command%(saveFN%index,lr,wd,rp,bs,mu)))
                        index = index + 1

def DSNSettingRP(modelName=modelName,newLoss=False):
    saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    command = '--val --save %s -r %f --coefL2 %f --rp %f -b %d -t 5 -lrd %f --trsize %d --tesize %d'
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
    index = 256 # index starts at 256 to 262
    notches = [100,200,400,800,2500,4000,7000]
    lr = 0.1; wd = 0.05; rp = 0.05; bs = 400; lrd = 5e-5; # based on the best model setting
    for n in notches:
        name = '%s%d' % (modelName,n)
        with open(name+'.pbs', 'w') as f:
            f.write(pbs_file % (name,command%(saveFN%index,lr,wd,rp,bs,lrd,n,2000)))
        index = index + 1

def genOld():
    path = 'pbsFiles/old_loss/'
    owd = os.getcwd()
    os.system('mkdir -p ' + path)
    os.chdir(path)
    os.system('echo generate pbs files in: ; pwd')
    # produce files
    torchModel()
    torchModelNoRP()
    torchModelLrd()
    DSNSetting()
    torchModelNoWD()
    DSNSettingRP()
    # change back path
    os.chdir(owd)

def genNew():
    path = 'pbsFiles/new_loss/'
    owd = os.getcwd()
    os.system('mkdir -p ' + path)
    os.chdir(path)
    os.system('echo generate pbs files in: ; pwd')
    # produce files
    torchModel(modelName=modelName,newLoss=True)
    torchModelNoRP(modelName=modelName,newLoss=True)
    torchModelLrd(modelName=modelName,newLoss=True)
    DSNSetting(modelName=modelName,newLoss=True)
    torchModelNoWD(modelName=modelName,newLoss=True)
    DSNSettingRP(modelName=modelName,newLoss=True)
    # change back path
    os.chdir(owd)

if __name__ == '__main__':
    genOld()
    genNew()

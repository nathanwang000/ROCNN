import os
# this version is for grid search in rorigid, so that unneccessary search are ommited
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
    command = "--val --full --save %s -r %f --coefL2 %f -b %d -m %f -t 5 -lrd %f"
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
    lrs = [1e-1,1e-2,1e-3,1e-4];
    batchSizes = [400]
    wds = [5e-1,5e-2,5e-3,0];
    lrds = [5e-5,5e-4,5e-3,5e-2]
    mus = [0,0.5,0.9];
    # archetectural changes: manually set
    # filterSizes = [5,10,15,20,25];
    # nFilters = [20,50,80];
    # poolSizes = [2,3,4];
    index = 1
    for lr in lrs:
        for bs in batchSizes:
            for wd in wds:
                for mu in mus:
                    for lrd in lrds:
                        name = '%s%d' % (modelName,index)
                        with open(name+'.pbs', 'w') as f:
                            f.write(pbs_file % (name,command%(saveFN%index,lr,wd,bs,mu,lrd)))
                        index = index + 1

def DSNSetting(modelName="LearningCurve",newLoss=False):
    # for learning curve
    saveFN = '/u/sciteam/wang12/scratch/torch_logs/'+modelName+'/log%d'
    command = '--val --save %s -r %f --coefL2 %f --rp %f -b %d -t 5 -lrd %f --trsize %d --tesize %d'
    if newLoss: 
        command = command + ' --rpLoss "variance/sum(w^2)"'
    index = 1
    notches = [100,200,400,800,2500,4000,7000]
    lr = 0.01; wd = 0.005; rp = 0; bs = 100; lrd = 5e-5; # based on the best model setting
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
    DSNSetting(modelName="LC")
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
    DSNSetting(modelName="LC",newLoss=True)
    # change back path
    os.chdir(owd)

if __name__ == '__main__':
    genOld()
    genNew()

import optparse
import cPickle as pickle

import sgd as optimizer
from rntn import RNTN
from rnn2deep import RNN2
from rnn import RNN
#from dcnn import DCNN
from rnn_changed import RNN3
import tree as tr
import time
import matplotlib.pyplot as plt
import numpy as np
import pdb


# This is the main training function of the codebase. You are intended to run this function via command line 
# or by ./run.sh

# You should update run.sh accordingly before you run it!


# TODO:
# Create your plots here

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)


    parser.add_option("--middleDim",dest="middleDim",type="int",default=10)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)

    # for DCNN only
    parser.add_option("--ktop",dest="ktop",type="int",default=5)
    parser.add_option("--m1",dest="m1",type="int",default=10)
    parser.add_option("--m2",dest="m2",type="int",default=7)
    parser.add_option("--n1",dest="n1",type="int",default=6)
    parser.add_option("--n2",dest="n2",type="int",default=12)
    
    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="RNN")

    (opts,args)=parser.parse_args(args)


    # make this false if you dont care about your accuracies per epoch, makes things faster!
    evaluate_accuracy_while_training = False

    # Testing
    if opts.test:
        test(opts.inFile,opts.data,opts.model)
        return
    
    print "Loading data..."
    train_accuracies = []
    dev_accuracies = []
    # load training data
    trees = tr.loadTrees('train')
    opts.numWords = len(tr.loadWordMap())

    if (opts.model=='RNTN'):
        nn = RNTN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN'):
        nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN2'):
        nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN3'):
        nn = RNN3(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='DCNN'):
        nn = DCNN(opts.wvecDim,opts.ktop,opts.m1,opts.m2, opts.n1, opts.n2,0, opts.outputDim,opts.numWords, 2, opts.minibatch,rho=1e-4)
        trees = cnn.tree2matrix(trees)
    else:
        raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN'%opts.model
    
    nn.initParams()

    sgd = optimizer.SGD(nn,alpha=opts.step,minibatch=opts.minibatch,
        optimizer=opts.optimizer)


    dev_trees = tr.loadTrees("dev")
    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d"%e
        sgd.run(trees)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(sgd.costt,fid)
            nn.toFile(fid)
        if evaluate_accuracy_while_training:
            print "testing on training set real quick"
            train_accuracies.append(test(opts.outFile,"train",opts.model,trees))
            print "testing on dev set real quick"
            dev_accuracies.append(test(opts.outFile,"dev",opts.model,dev_trees))
            # clear the fprop flags in trees and dev_trees
            for tree in trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            for tree in dev_trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            print "fprop in trees cleared"


    if evaluate_accuracy_while_training:
        # pdb.set_trace()
        # print train_accuracies
        # print dev_accuracies
        # TODO:
        # Plot train/dev_accuracies here?
        x = range(opts.epochs)
        plt.figure(figsize=(6,4))
        plt.plot(x, train_accuracies, color='b', marker='o', linestyle='-', label="training")
        plt.plot(x, dev_accuracies, color='g', marker='o', linestyle='-', label="dev")
        plt.title("Accuracy vs num epochs for wvecdim = 30.")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        #ylim(ymin=0, ymax=max(1.1*max(train_accuracies),3*min(train_accuracies)))
        plt.legend()
        plt.savefig("train_dev_acc_rnn2.png")

def test(netFile,dataSet, model='RNN', trees=None):
    if trees==None:
        trees = tr.loadTrees(dataSet)
    assert netFile is not None, "Must give model to test"
    print "Testing netFile %s"%netFile
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        
        if (model=='RNTN'):
            nn = RNTN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='RNN'):
            nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='RNN2'):
            nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(opts.model=='RNN3'):
            nn = RNN3(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='DCNN'):
            nn = DCNN(opts.wvecDim,opts.ktop,opts.m1,opts.m2, opts.n1, opts.n2,0, opts.outputDim,opts.numWords, 2, opts.minibatch,rho=1e-4)
            trees = cnn.tree2matrix(trees)
        else:
            raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN'%opts.model
        
        nn.initParams()
        nn.fromFile(fid)

    print "Testing %s..."%model

    cost,correct, guess, total = nn.costAndGrad(trees,test=True)
    correct_sum = 0
    for i in xrange(0,len(correct)):        
        correct_sum+=(guess[i]==correct[i])
    
    # TODO
    # Plot the confusion matrix?
    conf_arr = np.zeros((5,5))
    for i in xrange(0,len(correct)):
        current_correct = correct[i]
        current_guess = guess[i]
        conf_arr[current_correct][current_guess] += 1.0

    makeconf(conf_arr, model, dataSet)
    
    
    print "Cost %f, Acc %f"%(cost,correct_sum/float(total))
    return correct_sum/float(total)


def makeconf(conf_arr, model, dataSet):
    # makes a confusion matrix plot when provided a matrix conf_arr
    # model and dataset just for pathname to save figure
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    plt.savefig("conf_matrix_" + dataSet + "_" + model + ".png")

    #plt.show()


if __name__=='__main__':
    run()



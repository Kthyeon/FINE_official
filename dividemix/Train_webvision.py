from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import tqdm

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='',type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='./dataset/', type=str, help='path to dataset')

parser.add_argument('--distill', default=None, type=str, help='use "dynamic" for robust training')
parser.add_argument('--distill_mode', type=str, default='fine-gmm', choices=['kmeans','fine-kmeans','fine-gmm'], help='mode for distillation kmeans or eigen.')
parser.add_argument('--refinement', action='store_true', help='use refined label if in teacher_idx')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t Labeled loss: %.2f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)   
        
        #penalty = conf_penalty(outputs)
        L = loss #+ penalty      

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        
        
def test(epoch,net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)                 
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    return accs


def eval_train(model,all_loss):    
    model.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    losses = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]       
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' %(batch_idx,num_iter)) 
            sys.stdout.flush()    
                                    
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.cuda()
    return model

def get_singular_vector(features, labels):
    '''
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    '''
    
    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            _, _, v = np.linalg.svd(features[labels==index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict    
    
def get_features(model, dataloader):
    '''
    Concatenate the hidden features and corresponding labels 
    '''
    labels = np.empty((0,))

    model.eval()
    model.cuda()
    paths = []
    with tqdm(dataloader) as progress:
        for batch_idx, (data, label, path) in enumerate(progress):
            data, label = data.cuda(), label.long()
            
            feature = model.maxpool(model.relu(model.bn1(model.conv1(data))))
            feature = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(feature)))))
            feature = feature.view(feature.size(0), -1)
            
            for b in range(data.size(0)):
                paths.append(path[b])
                
            labels = np.concatenate((labels, label.cpu()))
            if batch_idx == 0:
                features = feature.detach().cpu()
            else:
                features = np.concatenate((features, feature.detach().cpu()), axis=0)
    
    return features, labels, paths

def get_score(singular_vector_dict, features, labels, normalization=True):
    '''
    Calculate the score providing the degree of showing whether the data is clean or not.
    '''
    if normalization:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat/np.linalg.norm(feat))) for indx, feat in enumerate(tqdm(features))]
    else:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat)) for indx, feat in enumerate(tqdm(features))]    
    
    return np.array(scores)
    
def fit_mixture(scores, labels, p_threshold=0.5):
    '''
    Assume the distribution of scores: bimodal gaussian mixture model
    
    return clean labels
    that belongs to the clean cluster by fitting the score distribution to GMM
    '''
    
    clean_labels = []
    indexes = np.array(range(len(scores)))
    probs = {}
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)
        
        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:,gmm.means_.argmax()]
        for i in range(len(cls_index)):
            probs[cls_index[i]] = prob[i]
#         weights, means, covars = g.weights_, g.means_, g.covariances_
        
#         # boundary? QDA!
#         a, b = (1/2) * ((1/covars[0]) - (1/covars[1])), -(means[0]/covars[0]) + (means[1]/covars[1])
#         c = (1/2) * ((np.square(means[0])/covars[0]) - (np.square(means[1])/covars[1]))
#         c -= np.log((weights[0])/np.sqrt(2*np.pi*covars[0]))
#         c += np.log((weights[1])/np.sqrt(2*np.pi*covars[1]))
#         d = b**2 - 4*a*c
        
#         bound = estimate_purity(feats, means, covars, weights)
        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold] 
    
    return np.array(clean_labels, dtype=np.int64), probs
    
    
def fine(current_features, current_labels, fit = 'kmeans', prev_features=None, prev_labels=None, p_threshold=0.7):
    '''
    prev_features, prev_labels: data from the previous round
    current_features, current_labels: current round's data
    
    return clean labels
    
    if you insert the prev_features and prev_labels to None,
    the algorthm divides the data based on the current labels and current features
    
    '''
    if (prev_features != None) and (prev_labels != None):
        singular_vector_dict = get_singular_vector(prev_features, prev_labels)
    else:
        singular_vector_dict = get_singular_vector(current_features, current_labels)

    scores = get_score(singular_vector_dict, features = current_features, labels = current_labels)
    
    if 'kmeans' in fit:
        clean_labels = cleansing(scores, current_labels)
        probs = None
    elif 'gmm' in fit:
        # fit a two-component GMM to the loss
        clean_labels, probs = fit_mixture(scores, current_labels, p_threshold)
    else:
        raise NotImplemented
    
    return clean_labels, probs

def cleansing(scores, labels):
    '''
    Assume the distribution of scores: bimodal spherical distribution.
    
    return clean labels 
    that belongs to the clean cluster made by the KMeans algorithm
    '''
    
    indexes = np.array(range(len(scores)))
    clean_labels = []
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
        if np.mean(scores[cls_index][kmeans.labels_==0]) < np.mean(scores[cls_index][kmeans.labels_==1]): kmeans.labels_ = 1 - kmeans.labels_
            
        clean_labels += cls_index[kmeans.labels_ == 0].tolist()
        
    return np.array(clean_labels, dtype=np.int64)
    
def extract_cleanidx(model, loader, mode='fine-kmeans', p_threshold=0.6):
    model.eval()
    for params in model.parameters(): params.requires_grad = False
        
    # get teacher_idx
    if 'fine' in mode:
        features, labels, paths = get_features(model, loader)
        teacher_idx, probs = fine(current_features=features, current_labels=labels, fit = mode, p_threshold=p_threshold)
    else: # get teacher _idx via kmeans
        teacher_idx = get_loss_list(model, loader)
        probs = None
        
    for params in model.parameters(): params.requires_grad = True
    model.train()
    
    teacher_idx = torch.tensor(teacher_idx)
    return teacher_idx, probs, paths






stats_log=open('./checkpoint/%s'%(args.id)+'_stats.txt','w') 
test_log=open('./checkpoint/%s'%(args.id)+'_acc.txt','w')     

warm_up=1

loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=5,root_dir=args.data_path,log=stats_log, num_class=args.num_class)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)



for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr              
    eval_loader = loader.run('eval_train')  
    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')   
    imagenet_acc = test(0,net1,net2,imagenet_valloader)
    print(imagenet_acc)

    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:     
        eval_loader = loader.run('eval_train')
        teacher_idx_1, prob1_dict, paths1 = extract_cleanidx(net1, eval_loader, mode=args.distill_mode, p_threshold=args.p_threshold)
        eval_loader = loader.run('eval_train')
        teacher_idx_2, prob2_dict, paths2 = extract_cleanidx(net2, eval_loader, mode=args.distill_mode, p_threshold=args.p_threshold)
        if data_num == None:
            data_num = len(prob1_dict.keys())
        pred1, pred2 = np.zeros(data_num, dtype=bool), np.zeros(data_num, dtype=bool)
        prob1, prob2 = np.zeros(data_num), np.zeros(data_num)

        for index in teacher_idx_1:
            pred1[index] = True
        for index in teacher_idx_2:
            pred2[index] = True

        for i in range(data_num):
            prob1[i] = prob1_dict[i]
            prob2[i] = prob2_dict[i]     
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2    

    
    web_acc = test(epoch,net1,net2,web_valloader)  
    imagenet_acc = test(epoch,net1,net2,imagenet_valloader)  
    
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))  
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.flush()  
       
    print('\n==== net 1 evaluate training data loss ====') 
    prob1,all_loss[0]=eval_train(net1,all_loss[0])   
    print('\n==== net 2 evaluate training data loss ====') 
    prob2,all_loss[1]=eval_train(net2,all_loss[1])
    torch.save(all_loss,'./checkpoint/%s.pth.tar'%(args.id))   
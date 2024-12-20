#author: akshitac8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import copy
import pdb
import scipy.io
from sklearn.decomposition import PCA
# from config import opt
# from KNN import knn_classifier_cuda
from sklearn.neighbors import KNeighborsClassifier

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5,
                 _nepoch=20, _batch_size=100, generalized=True, final_classifier=None,
                 netFR=None, dec_size=4096, dec_hidden_size=4096, opt=None):
        self.opt = opt
        self.train_X = _train_X.clone()
        self.train_Y = _train_Y.clone() 
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.netFR = netFR
        if self.netFR:
            self.netFR.eval()
            if self.opt == None:
                self.input_dim = self.input_dim + dec_hidden_size + dec_size
            elif self.opt.feature_component == 'FR':
                self.input_dim = self.input_dim
            elif self.opt.feature_component == 'FR_h':
                self.input_dim = self.input_dim + dec_hidden_size
            elif self.opt.feature_component == 'FR_h_a':
                self.input_dim = self.input_dim + dec_hidden_size + dec_size

            self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.train_X = self.compute_fear_out(self.train_X, self.input_dim)
            test_unseen_feature = self.compute_fear_out(self.test_unseen_feature, self.input_dim)
            test_seen_feature = self.compute_fear_out(self.test_seen_feature, self.input_dim)
            test_seen_feature = np.array(test_seen_feature)
            test_unseen_feature = np.array(test_unseen_feature)
            self.test_unseen_feature = self.compute_fear_out(self.test_unseen_feature, self.input_dim)
            self.test_seen_feature = self.compute_fear_out(self.test_seen_feature, self.input_dim)

        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            # if opt.final_classifier == 'softmax':
            if final_classifier == 'softmax':
                self.acc_seen, self.acc_unseen, self.H, self.epoch= self.fit()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
            
            # elif opt.final_classifier == 'knn':
            elif final_classifier == 'knn':
                #raise Exception
                # clf = knn_classifier_cuda(tr_X=self.train_X, tr_Y=self.train_Y, n_neighbors=opt.k_nn)
                # self.acc_seen = clf.predict(te_X=self.test_seen_feature, te_Y=self.test_seen_label)
                # self.acc_unseen = clf.predict(te_X=self.test_unseen_feature, te_Y=self.test_unseen_label)
                # self.H = 2 * self.acc_seen * self.acc_unseen / (self.acc_seen + self.acc_unseen)
                
                clf = KNeighborsClassifier(n_neighbors=opt.k_nn)
                clf.fit(X=self.train_X, y=self.train_Y)
                pred_Y_s = torch.from_numpy(clf.predict(self.test_seen_feature))
                pred_Y_u = torch.from_numpy(clf.predict(self.test_unseen_feature))
                #print("prediction labels shapes", pred_Y_u.shape, pred_Y_s.shape)
                pred_y_s = np.array(pred_Y_s)
                pred_y_u = np.array(pred_Y_u)
                #scipy.io.savemat('test_seen_and_unseen_labels_PoFE.mat', {'predicted_test_unseen_labels':pred_y_u, 'predicted_test_seen_labels':pred_y_s})
                self.acc_seen = self.compute_per_class_acc_gzsl_knn(pred_Y_s, self.test_seen_label,  self.seenclasses)
                self.acc_unseen = self.compute_per_class_acc_gzsl_knn( pred_Y_u, self.test_unseen_label,  self.unseenclasses)
                self.H = 2 * self.acc_seen * self.acc_unseen / (self.acc_seen + self.acc_unseen)
            else:
                print("classifier is not existing")
            
            
        else:
            # self.acc,self.best_model = self.fit_zsl() 
            # if opt.final_classifier == 'softmax':
            if final_classifier == 'softmax':
                self.acc,self.best_model = self.fit_zsl() 
            # elif opt.final_classifier == 'knn':
            elif final_classifier == 'knn':
                #raise Exception
                ### clf = knn_classifier_cuda(tr_X=self.train_X, tr_Y=self.train_Y, n_neighbors=opt.k_nn)
                ### self.acc = clf.predict(te_X=self.test_unseen_feature, te_Y=self.test_unseen_label)
                zsl_clf = KNeighborsClassifier(n_neighbors=opt.k_nn)
                zsl_clf.fit(X=self.train_X, y=self.train_Y)
                pred_Y_u = torch.from_numpy(zsl_clf.predict(self.test_unseen_feature))
                tul = np.array(pred_Y_u)
                #scipy.io.savemat('synthesized_unseen_plus_Post_FE.mat', {'test_unseen_features':test_unseen_feature, 'test_unseen_labels':tul})
                self.acc = self.compute_per_class_acc_knn(pred_Y_u, util.map_label(self.test_unseen_label, self.unseenclasses), self.unseenclasses.size(0))
            else:
                print("classifier is not existing")
            
            
            
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                #mean_loss += loss.data[0]
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
            acc, op, pl = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            #print('acc %.4f' % (acc))
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(self.model.state_dict())
                #scipy.io.savemat('test_unseen_features_CZSL.mat', {'test_unseen_features':op, 'test_unseen_labels':pl})
        return best_acc, best_model 
        
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = 0
            acc_unseen = 0
            acc_seen, op, pl = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            #scipy.io.savemat('test_seen_features_GZSL.mat', {'test_seen_features':op, 'test_seen_labels':pl})
            acc_unseen, op, pl = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            #acc_seen, op, pl = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            #scipy.io.savemat('test_unseen_features_GZSL.mat', {'test_unseen_features':op, 'test_unseen_labels':pl})
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                #scipy.io.savemat('test_seen_features_GZSL.mat', {'test_seen_features':op, 'test_seen_labels':pl})
                #scipy.io.savemat('test_unseen_features_GZSL.mat', {'test_unseen_features':op, 'test_unseen_labels':pl})
        return best_seen, best_unseen, best_H,epoch
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            output = self.model(inputX)
            op = output
            op = op.detach().cpu().numpy()
            op = np.array(op)  
            _, predicted_label[start:end] = torch.max(output.data, 1)
            pl = predicted_label
            pl = pl.detach().cpu().numpy()
            pl = np.array(pl)
            start = end
        op = test_X
        op = op.cuda()
        op = self.model(op)
        op = op.cpu().detach().numpy()
        op = np.array(op)
        #print("op pl shapes", op.shape, pl.shape)
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc, op, pl

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            output = self.model(inputX)
            op = output
            op = op.detach().cpu().numpy()
            op = np.array(op) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            pl = predicted_label
            pl = pl.detach().cpu().numpy()
            pl = np.array(pl) 
            start = end
        op = test_X
        op = op.cuda()
        op = self.model(op)
        op = op.cpu().detach().numpy()
        op = np.array(op)
        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc, op, pl

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
        return acc_per_class.mean() 


    def compute_fear_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            _,_,_,_, _, feat2 = self.netFR(inputX)
            feat1 = self.netFR.getLayersOutDet()

            if self.opt == None:
                new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            elif self.opt.feature_component == 'FR':
                new_test_X[start:end] = inputX.data.cpu()
            elif self.opt.feature_component == 'FR_h':
                new_test_X[start:end] = torch.cat([inputX,feat1],dim=1).data.cpu()
            elif self.opt.feature_component == 'FR_h_a':
                new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            
            start = end
            # pca = PCA(n_components=4096,whiten=False)
            # fit = pca.fit(new_test_X)
            # features = pca.fit_transform(new_test_X)
            # features = torch.from_numpy(features)
            # fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
            # new_test_X = features.div(fnorm.expand_as(features))
        #print(new_test_X.shape)
        return new_test_X

    def compute_per_class_acc_gzsl_knn( self,  predicted_label, test_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (predicted_label == i)
            if torch.sum(idx)==0:
                acc_per_class +=0
            else:
                acc_per_class += float(torch.sum(predicted_label[idx] == test_label[idx])) / float(torch.sum(idx))
        acc_per_class /= float(target_classes.size(0))
        return acc_per_class
        
    def compute_per_class_acc_knn(self, predicted_label, test_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx)==0:
                acc_per_class +=0
            else:
                acc_per_class += torch.sum(predicted_label[idx]==test_label[idx]).float() / torch.sum(idx)
        acc_per_class /= float(nclass)
        return acc_per_class.mean() 
    
class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o

import argparse
import random,os,sys
import numpy as np
import csv
from sklearn.decomposition import PCA
import pandas as pd
import keras.backend as K
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
#from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam,SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from model import MultiOmicsDrugGraphResNetModel
import hickle as hkl
import scipy.sparse as sp
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
model_suffix = "My_Model_DRNCDR"


TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']

'''
--------------Loading Dataset------------------

'''

DPATH = 'D:/PROGRAMS/DNN/DRNCDR/data'
Drug_info_file = '%s/Drug_annotation.csv'%DPATH
Cell_line_info_file = '%s/Cell_lines_annotations.txt'%DPATH
Drug_feature_file = '%s/drug_graph_feat'%DPATH
Genomic_mutation_file = '%s/mutation.csv'%DPATH
Cancer_response_exp_file = '%s/GDSC_IC50.csv'%DPATH
Gene_expression_file = '%s/gene_expression.csv'%DPATH
Methylation_file = '%s/methylation.csv'%DPATH
Max_atoms = 100


'''
----------------Collecting Drug IDs and Cell line IDs from info files-------------
'''

def IDGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,filtered):
    #drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}

    #map cellline --> cancer type
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        #if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    #load demap cell lines genomic mutation features
    mutation_feature = pd.read_csv(Genomic_mutation_file,sep=',',header=0,index_col=[0])
    cell_line_id_set = list(mutation_feature.index)

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    
    #only keep overlapped cell lines
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    
    #load methylation 
    methylation_feature = pd.read_csv(Methylation_file,sep=',',header=0,index_col=[0])
    #load Copy Number
    #copynumber_feature = pd.read_csv(CopyNumber_file,sep=',',header=0,index_col=[0])
    assert methylation_feature.shape[0]==gexpr_feature.shape[0]==mutation_feature.shape[0]        
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    #filter experiment data
    drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    
    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                    data_idx.append((each_cellline,pubchem_id,ln_IC50,cellline2cancertype[each_cellline])) 
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx

#split into training and test set
def DataSplit(data_idx,ratio = 0.10):
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    print(len(data_train_idx))
    return data_train_idx,data_test_idx


"""-----------------------------Drug wise info collection and feature calculation-------------"""

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='uint8')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='uint8')      
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]


'''-------------------Extraction of Drug and Cell Line Features---------------'''

def FeatureSelection(data_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_mutation_feature = mutation_feature.shape[1]
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_methylation_features = methylation_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    mutation_data = np.zeros((nb_instance, nb_mutation_feature),dtype='uint8')
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='uint8') 
    methylation_data = np.zeros((nb_instance, nb_methylation_features),dtype='uint8') 
    target = np.zeros(nb_instance,dtype='uint8')
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,ln_IC50,cancer_type = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        mutation_data[idx,:] = mutation_feature.loc[cell_line_id].values
        gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id].values
        methylation_data[idx,:] = methylation_feature.loc[cell_line_id].values
        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type,cell_line_id,pubchem_id])
    return drug_data,mutation_data,gexpr_data,methylation_data,target,cancer_type_list
    


class MyCallback(Callback):
    def __init__(self,validation_data,patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('D:/PROGRAMS/DNN/DRNCDR/MyDRN-CDR_%s.h5'%model_suffix)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        pcc_val = pearsonr(self.y_val, y_pred_val[:,0])[0]
        print('pcc-val: %s' % str(round(pcc_val,4)))
        if pcc_val > self.best:
            self.best = pcc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return
        



def ModelTraining(model,X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,validation_data,nb_epoch=10):
    optimizer = Adam(learning_rate=0.001, amsgrad=False)
    model.compile(optimizer = optimizer,loss='mean_squared_error',metrics=['mse','accuracy'])
    model.summary()
    #EarlyStopping(monitor='val_loss',patience=5)
    callbacks = [ModelCheckpoint('D:/PROGRAMS/DNN/DRNCDR/best_DRNCDRmodel_%s.h5'%model_suffix,monitor='val_loss',save_best_only=False, save_weights_only=False),
                MyCallback(validation_data=validation_data,patience=10)]
    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
    #validation data

    model1=model.fit(x=[X_drug_feat_data_train,X_drug_adj_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train],y=Y_train,batch_size=64,epochs=nb_epoch,validation_split=0,callbacks=callbacks)
    plt.plot(model1.history['loss'])
    #plt.plot(hmodel1.history['val_accuracy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig('plot-trial-ResNet.png')
    return model


def ModelEvaluate(model,X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list,data_test_idx_current,Y_val):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    
    Y_pred = model.predict([X_drug_feat_data_test,X_drug_adj_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test])
    overall_pcc = pearsonr(Y_pred[:,0],Y_test)[0]
    '''Y_pred.tofile('prediction_results.csv',sep=',',format='%10.5f')'''
    print("The overall Pearson's correlation is %.4f."%overall_pcc)

    r2 = r2_score(Y_pred,Y_test)
    print(f"Coefficient of Determination: {r2}")

    #Writing predictions to a file
    data = []
    for i in range(len(data_test_idx_current)):
        drug_ = data_test_idx_current[i][1]
        cellline_ = data_test_idx_current[i][0]
        predicted_ = Y_pred[i,0]
        true_ = Y_test[i]
        data.append([drug_,cellline_,true_,predicted_])
    df = pd.DataFrame(data, columns=['Drug_ID','Cell_Line_ID', 'Original_IC50', 'Predicted_IC50'])
    # Write the DataFrame to a CSV file
    csv_filename = 'D:/PROGRAMS/DNN/DRNCDR/predicted_ic50_values.csv'
    df.to_csv(csv_filename, index=False)

'''----------------------------Model Traning and Evaluation----------------------'''

def main():
    random.seed(0)
    mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx = IDGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,False)
    data_train_idx,data_test_idx = DataSplit(data_idx)
    #Extract features for training and test 
    X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,cancer_type_train_list = FeatureSelection(data_train_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature)
    X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list = FeatureSelection(data_test_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature)
    
    #X_mutation_data_train=X_mutation_data_train.T
    #X_mutation_data_test=X_mutation_data_test.T

    print("mutation shape",X_mutation_data_train.shape)
    pca_mutation = PCA(n_components=100)
    pca_gexpr = PCA(n_components=100)
    pca_methy = PCA(n_components=100)

    #mutation_input_reduced = pca_mutation.fit_transform(X_mutation_data_train)
    gexpr_input_reduced = pca_gexpr.fit_transform(X_gexpr_data_train)
    methy_input_reduced = pca_methy.fit_transform(X_methylation_data_train)

    #mutation_input_reduced_test = pca_mutation.fit_transform(X_mutation_data_test)
    gexpr_input_reduced_test = pca_gexpr.fit_transform(X_gexpr_data_test)
    methy_input_reduced_test = pca_methy.fit_transform(X_methylation_data_test)

    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  
    
    validation_data = [[X_drug_feat_data_test,X_drug_adj_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test],Y_test]
    model = MultiOmicsDrugGraphResNetModel().FeatureExtract(X_drug_data_train[0][0].shape[-1],X_mutation_data_train.shape[-1],gexpr_input_reduced.shape[-1],methy_input_reduced.shape[-1],[256,256,256])
    print('Begin training...')
    model = ModelTraining(model,X_drug_data_train,X_mutation_data_train,gexpr_input_reduced,methy_input_reduced,Y_train,validation_data,nb_epoch=10)
    ModelEvaluate(model,X_drug_data_test,X_mutation_data_test,gexpr_input_reduced_test,methy_input_reduced_test,Y_test,cancer_type_test_list,'%s/DeepCDR_%s.log'%(DPATH,model_suffix),data_test_idx,Y_test)

if __name__=='__main__':
    main()
    


import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from sklearn.decomposition import PCA

#implementing GraphLayer
class GraphLayer(keras.layers.Layer):

    def __init__(self, step_num=1, activation=None, **kwargs):
        self.step_num = step_num
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        config = {'step_num': self.step_num, 'activation': keras.activations.serialize(self.activation)}
        return {**super().get_config(), **config}

    def _get_walked_edges(self, edges, step_num): # to get the adjacency matrix of graphs
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(K.dot(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return K.cast(K.greater(deeper, 0.0), K.floatx())

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        return self.activation(self._call(features, edges))

    def _call(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')

#implementing Graph Convolution
class GraphConv(GraphLayer):

    def __init__(self, units, kernel_initializer='glorot_uniform', use_bias=True, **kwargs):
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.W, self.b = None, None
        super().__init__(**kwargs)

    def get_config(self):
        config = {'units': self.units, 'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
                  'use_bias': self.use_bias}
        return {**super().get_config(), **config}

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(feature_dim, self.units), initializer=self.kernel_initializer, name='W')
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,), initializer='zeros', name='b')
        super().build(input_shape)

    def _call(self, features, edges):
        features = K.dot(features, self.W) + (self.b if self.use_bias else 0)
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features)
      
        
class MultiOmicsDrugGraphResNetModel(object):
    def __init__(self,regr=True):
        self.regr = regr
    def Unit(x, filters,  pool=False):
        res = x
        if pool:
            x = MaxPooling2D(pool_size=(1, 2), padding="same")(x)
            res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(1, 2), padding="same")(res)
        out = BatchNormalization()(x)
        out = Activation("relu")(out)
        out = Conv2D(filters=filters, kernel_size=[1, 150], strides=[1, 1], padding="same")(out)

        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv2D(filters=filters, kernel_size=[1, 150], strides=[1, 1], padding="same")(out)

        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv2D(filters=filters, kernel_size=[1, 150], strides=[1, 1], padding="same")(out)


        out = keras.layers.add([res, out])

        return out

    def FeatureExtract(self,drug_dim,mutation_dim,gexpr_dim,methy_dim,units_list):
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')#drug_dim=75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')
        
        mutation_input = Input(shape=(1,mutation_dim,1),name='mutation_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        methy_input = Input(shape=(methy_dim,),name='methy_feat_input')
        #drug feature with GCN
        GCN_layer = GraphConv(units=units_list[0],step_num=1)([drug_feat_input,drug_adj_input])
        GCN_layer = Activation('relu')(GCN_layer)
        GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        
        for i in range(len(units_list)-1):
            GCN_layer = GraphConv(units=units_list[i+1],step_num=1)([GCN_layer,drug_adj_input])
            GCN_layer = Activation('relu')(GCN_layer)
            GCN_layer = BatchNormalization()(GCN_layer)
            GCN_layer = Dropout(0.1)(GCN_layer)
            
        GCN_layer = GraphConv(units=100,step_num=1)([GCN_layer,drug_adj_input])
        GCN_layer = Activation('relu')(GCN_layer)
        GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        #global pooling
        x_drug = GlobalMaxPooling1D()(GCN_layer)


        #genomic mutation feature 
        x_mut = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(mutation_input)
        x_mut = MaxPooling2D(pool_size=(1,5))(x_mut)
        x_mut = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_mut)
        x_mut = MaxPooling2D(pool_size=(1,10))(x_mut)
        x_mut = Flatten()(x_mut)
        x_mut = Dense(100,activation = 'relu')(x_mut)
        x_mut = Dropout(0.1)(x_mut)
        #gexp feature
        x_gexpr = Dense(256)(gexpr_input)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation='relu')(x_gexpr)
        #methylation feature
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        x_methy = Dropout(0.1)(x_methy)
        x_methy = Dense(100,activation='relu')(x_methy)

        x = x_drug
        x = Concatenate()([x,x_mut])#concat layers
        x = Concatenate()([x,x_gexpr])
        x = Concatenate()([x,x_methy])
            #x = Concatenate()([x_mut,x_drug,x_gexpr,x_methy])
        x = Dense(400,activation = 'tanh')(x)
        #x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        # trial
        x = Conv2D(filters=32, kernel_size=[1, 150], strides=[1, 1], padding="valid")(x)
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 32)
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 32)
        #x = KerasMultiSourceGCNModel.Unit(x, 30)        
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 64,pool=True)
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 64)
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 64)
        #x = KerasMultiSourceGCNModel.Unit(x, 10)
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 128,pool=True)
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 128)
        x = MultiOmicsDrugGraphResNetModel.Unit(x, 128)
        #x = KerasMultiSourceGCNModel.Unit(x, 5)

        #x = AveragePooling2D(pool_size=(1, 2), padding='same')(x)
        
        print(x.shape)
        x = Flatten()(x)
        #x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[drug_feat_input,drug_adj_input,mutation_input,gexpr_input,methy_input],outputs=output)
        return model    

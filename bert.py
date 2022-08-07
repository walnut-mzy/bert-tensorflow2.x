import tensorflow as tf
from model.Layer.transformer import Transformer
from model.Layer.Embedding import EmbeddingProcessor
from tensorflow.keras.layers import *
from tensorflow.keras import *
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertModel,BertConfig
import pandas as pd
from tqdm import *
import setting
from transformers import TFBertForSequenceClassification



# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
class Bert(tf.keras.Model):


    def __init__(self, config, **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.vocab_size = config['Vocab_Size']
        self.embedding_size = config['Embedding_Size']
        self.max_seq_len = config['Max_Sequence_Length']
        self.segment_size = config['Segment_Size']
        self.num_transformer_layers = config['Num_Transformer_Layers']
        self.num_attention_heads = config['Num_Attention_Heads']
        self.intermediate_size = config['Intermediate_Size']
        self.initializer_range = config['Initializer_Variance']
        self.class_nums=config["class_nums"]
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.embedding = EmbeddingProcessor(vocab_szie=self.vocab_size, embedding_size=self.embedding_size,
                                            max_seq_len=self.max_seq_len,
                                            segment_size=self.segment_size, )
        self.transformer_blocks = [Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)] * self.num_transformer_layers
        # self.nsp_predictor = tf.keras.layers.Dense(2)
        self.dn1=tf.keras.layers.Dense(self.class_nums,bias_initializer=self.initializer,activation="softmax")

    def call(self, inputs, training=None):
        # inputs=inputs.shape
        batch_x=inputs["input_ids"]
        batch_mask=inputs["token_type_ids"]
        batch_segment=inputs["attention_mask"]
        x = self.embedding((batch_x, batch_segment))
        for i in range(self.num_transformer_layers):
            x = self.transformer_blocks[i](x, mask=batch_mask, training=training)


        first_token_tensor = x[:, 0, :]  # [batch_size ,hidden_size]
        # nsp_predict = self.nsp_predictor(first_token_tensor)
       # mlm_predict = tf.matmul(x, self.embedding.token_embedding.embeddings, transpose_b=True)

        sequence_output=self.dn1(first_token_tensor)
        return  sequence_output
import os
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
class Dataset:
    def __init__(self,tokenize,label):
        self.input_ids=tokenize["input_ids"]
        self.token_type_ids=tokenize["token_type_ids"]
        self.attention_mask=tokenize["attention_mask"]
        self.len_input_ids=len(self.input_ids)
        self.len_token_type_ids=len(self.token_type_ids)
        self.len_attention_mask=len(self.attention_mask)
        self.label=tf.one_hot(label,depth=2)
        self.len_label=len(label)

       # print(tf.convert_to_tensor(self.attention_mask).shape,tf.convert_to_tensor(self.label).shape,tf.convert_to_tensor(self.input_ids).shape,tf.convert_to_tensor(self.token_type_ids))

    # def __getitem__(self, index):
    #     return [self.input_ids[index],self.token_type_ids[index],self.attention_mask[index]],self.label[index]
    def make_dataset(self):
        #self.input_ids,self.token_type_ids,self.attention_mask
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "input_ids": self.input_ids,
                    "token_type_ids": self.token_type_ids,
                    "attention_mask": self.attention_mask,
                }, self.label

            )
        )
        dataset = dataset.shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch)
        print(dataset)
        return dataset
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    print(epoch)
    #logs如下
    """
    {'loss': 0.35939720273017883, 'accuracy': 0.5134039521217346, 'val_loss': 0.2836693525314331, 'val_accuracy': 0.476856529712677}
    """
    """
    写入loss等基本信息
    epch,loss,accuracy,val_loss,val_accuracy
    """
    self.fp.write(str(epoch)+","+str(logs["loss"])+","+str(logs["accuracy"])+","+str(logs["val_loss"])+","+str(logs["val_accuracy"])+"\n")


# self.vocab_size = config['Vocab_Size']
# self.embedding_size = config['Embedding_Size']
# self.max_seq_len = config['Max_Sequence_Length']
# self.segment_size = config['Segment_Size']
# self.num_transformer_layers = config['Num_Transformer_Layers']
# self.num_attention_heads = config['Num_Attention_Heads']
# self.intermediate_size = config['Intermediate_Size']
# self.initializer_range = config['Initializer_Variance']

if __name__ == '__main__':
    df = pd.read_csv('data/Dataset.csv')
    df.info()
    df['sentiment'].value_counts()
    x = list(df['review'])
    y = list(df['sentiment'])
    print("数据集数量：",len(x))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
    test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)
    print(train_encoding.keys())

    # # 可以这样查看词典
    """
    dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    """
    vocab = tokenizer.vocab

    dataset_train=Dataset(train_encoding,y_train).make_dataset()
    dataset_test=Dataset(test_encoding,y_test).make_dataset()
    Config = {
        'Character_Frequency_Threshold': 3,
        'Segment_Size': 2,
        'Batch_Size': setting.batch,
        'Max_Sequence_Length': 64,  # 最大长度
        'Mask_Rate': 0.15,
        'Vocab_Size':len(vocab),
        'Embedding_Size': 256,
        'Num_Transformer_Layers': 24,
        'Num_Attention_Heads': 16,
        'Intermediate_Size': 1024,
        'Initializer_Variance': 0.02,  # 权重初始化方差，默认0.02
        'class_nums': 2
    }
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    # model=Bert(config=Config)
    optimizer =tf.keras.optimizers.Adam(learning_rate=1e-5,epsilon=1e-08, clipnorm=1)
    train_loss = tf.keras.losses.CategoricalCrossentropy()
    train_accuracy =tf.keras.metrics.Accuracy(name='train_accuracy')

    @tf.function
    def train_one_step(x, y):
        """
        一次迭代过程
        """
        # 求loss
        with tf.GradientTape() as tape:
            predictions = model(x)

            predictions=predictions["logits"]

            train_accuracy.update_state(y_true=y, y_pred=predictions)
            loss1 = train_loss(y_true=y, y_pred=predictions)
        # 求梯度
        grad = tape.gradient(loss1, model.trainable_variables)
        # 梯度下降，更新噪声图片
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss1


    callbacks = [
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(setting.save_path, "bert{epoch}.h5"),
            monitor='val_loss',
            save_weights_only=True,
            verbose=10
        ),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                  patience=20,
        #                                  restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=2, embeddings_freq=0,
            embeddings_metadata=None,
        ),
        #myCallback()
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
    ]
    # 查看模型结构

    model.compile(
        optimizer=optimizer,
        loss=train_loss,
        metrics=['accuracy']
    )


    model.fit(
        dataset_train,
        epochs=setting.epoch,

        # initial_epoch=setting.initial_epoch,
        validation_data=dataset_test,

        callbacks=callbacks,
    )

    # for epoch in range(setting.epoch):
    #     # 使用tqdm提示训练进度
    #     with tqdm(total=len(dataset_train) / setting.batch,
    #               desc='Epoch {}/{}'.format(epoch, setting.epoch)) as pbar:
    #         # 每个epoch训练settings.STEPS_PER_EPOCH次
    #         for x1,x2,x3, y in dataset_train:
    #
    #             loss2 = train_one_step([x1,x2,x3], y)
    #             pbar.set_postfix(loss='%.4f' % float(loss2), acc=float(train_accuracy.result()))
    #             pbar.update(1)



# import dependencies
import pandas as pd
import numpy as np
import tensorflow as tf

# 데이터 불러오기
df=pd.read_csv('/Users/pio/work/house_price.csv')

# 데이터 전처리

# remodel_year ->리모델링한 후 기간
REMODEL_YEAR=2019-df['REMODEL_YEAR']
df['REMODEL_YEAR']=REMODEL_YEAR

# 양적 자료 로그로 변환

# df 내 BUILDING,LAND_AREA column은 로그 변환

BUILDING_AREA_log=np.log(df['BUILDING_AREA'])
LAND_AREA_log=np.log(df['LAND_AREA'])

df['BUILDING_AREA']=BUILDING_AREA_log
df['LAND_AREA']=LAND_AREA_log

# CONDITION -> 숫자값으로 변경

CON_to_NUM=[]
for i in range(len(df)) :
    if df['CONDITION'][i]=='Poor' :
        CON_to_NUM.append(0)
    elif df['CONDITION'][i]=='Fair' :
        CON_to_NUM.append(1)
    elif df['CONDITION'][i]=='Average' :
        CON_to_NUM.append(2)
    elif df['CONDITION'][i] == 'Good':
        CON_to_NUM.append(3)
    elif df['CONDITION'][i] == 'Very Good':
        CON_to_NUM.append(4)
    else : CON_to_NUM.append(5)

CON_to_NUM=np.array(CON_to_NUM)

df['CONDITION']=CON_to_NUM


# 양적자료와 질적자료, 라벨값 구분해놓기

## 원래 PRICE는 따로 PRICE_origin에 저장 /
PRICE_origin=pd.DataFrame(df['PRICE'])

df_quant=df[['BATHROOMS','ROOMS','BEDROOMS','REMODEL_YEAR',
             'STORIES','BUILDING_AREA','KITCHENS','LAND_AREA']]

df_cond=df[['CONDITION']]
df_bath=df[['BATHROOMS']]
df_num_units=df[['NUM_UNITS']]
df_fireplaces=df[['FIREPLACES']]

# Define Nods

# quantative NN

# input layer
X1=tf.placeholder('float32',shape=[None,8],name='quant_input')

# layer 1
W1=tf.get_variable('Weight_1',shape=[8,16],initializer=tf.contrib.layers.xavier_initializer())
B1=tf.get_variable('bias_1',shape=[16],initializer=tf.contrib.layers.xavier_initializer())

layer1=tf.matmul(X1,W1)+B1

# layer2
W2=tf.get_variable('Weight_2',shape=[16,32],initializer=tf.contrib.layers.xavier_initializer())
B2=tf.get_variable('bias_2',shape=[32],initializer=tf.contrib.layers.xavier_initializer())

layer2=tf.matmul(layer1,W2)+B2

# layer3
W3=tf.get_variable('Weight_3',shape=[32,1],initializer=tf.contrib.layers.xavier_initializer())
B3=tf.get_variable('bias_3',shape=[1],initializer=tf.contrib.layers.xavier_initializer())

layer3=tf.matmul(layer2,W3)+B3

# qualitative

# condition
cond_ph=tf.placeholder('int32',shape=[None,1],name='cond_ph')
cond_one_hot=tf.one_hot(cond_ph,6)
cond_one_hot=tf.reshape(cond_one_hot,[-1,6])

cond_W=tf.get_variable('Weight_cond',shape=[6,1],initializer=tf.contrib.layers.xavier_initializer())
cond_B=tf.get_variable('bias_cond',shape=[1,1],initializer=tf.contrib.layers.xavier_initializer())


# bathroom
bathroom_ph=tf.placeholder('int32',shape=[None,1],name='bathroom_ph')
bathroom_one_hot=tf.one_hot(bathroom_ph,13)
bathroom_one_hot=tf.reshape(bathroom_one_hot,[-1,13])

bathroom_W=tf.get_variable('Weight_bath',shape=[13,1],initializer=tf.contrib.layers.xavier_initializer())
bathroom_B=tf.get_variable('bias_bath',shape=[1,1],initializer=tf.contrib.layers.xavier_initializer())

# num_units
num_units_ph=tf.placeholder('int32',shape=[None,1],name='num_units_ph')
num_units_one_hot=tf.one_hot(num_units_ph,7)
num_units_one_hot=tf.reshape(num_units_one_hot,[-1,7])

num_units_W=tf.get_variable('Weight_num_unit',shape=[7,1],initializer=tf.contrib.layers.xavier_initializer())
num_units_B=tf.get_variable('bias_num_unit',shape=[1,1],initializer=tf.contrib.layers.xavier_initializer())

# fire_place
fire_place_ph=tf.placeholder('int32',shape=[None,1],name='fire_place_ph')
fire_place_one_hot=tf.one_hot(fire_place_ph,14)
fire_place_one_hot=tf.reshape(fire_place_one_hot,[-1,14])

fire_place_W=tf.get_variable('Weight_fire_place',shape=[14,1],initializer=tf.contrib.layers.xavier_initializer())
fire_place_B=tf.get_variable('bias_fire_place',shape=[1,1],initializer=tf.contrib.layers.xavier_initializer())

Y=tf.placeholder('float32',shape=[None,1],name='label')

# graph
Predicted=layer3 + \
          tf.add(tf.matmul(cond_one_hot,cond_W),cond_B) + \
          tf.add(tf.matmul(bathroom_one_hot,bathroom_W),bathroom_B) + \
          tf.add(tf.matmul(num_units_one_hot,num_units_W),num_units_B) +\
          tf.add(tf.matmul(fire_place_one_hot,fire_place_W),fire_place_B)

error_rate=tf.abs(Predicted-Y)/Y*100

cost = tf.reduce_mean(tf.square(Predicted - Y))


# train
train = tf.train.AdamOptimizer(1).minimize(cost)

# initializing the graph
sess=tf.Session()
sess.run(tf.global_variables_initializer())


# run

for step in range(2000001) :
    cost_val, Predicted_val, _ =sess.run([cost,Predicted,train],feed_dict={
        X1:df_quant,
        cond_ph:df_cond,
        bathroom_ph:df_bath,
        fire_place_ph:df_fireplaces,
        num_units_ph:df_num_units,
        Y:PRICE_origin})
    if step % 1000 == 0 :
        print(cost_val)


action = 0 if tf.random_uniform(shape=[1],minval=0,maxval=1,dtype=tf.float32) < probs[0][0] else 1
rand_0_1 = tf.random_uniform(shape=[1],minval=0,maxval=1,dtype=tf.float32)
action = tf.cast(tf.less(rand_0_1,probs[0][1]),dtype='float32')
'''
# ValueError: Cannot feed value of shape (33149,) for Tensor 'label:0', which has shape '(?, 1)' 뜰 경우
해당하는 텐서와 feed되는 데이터 확인
np.shape(x) 이용 - shape 확인 
np.reshape(x,()) 이용 - reshape 해주기

'''
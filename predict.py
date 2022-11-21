import json
import re
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

# from google.colab import drive
# drive.mount('/content/drive')

# %cd /content/drive/My\ Drive/Colab\ Notebooks/QA_HDH/CNN_tfidf

start = time.time()

url_datasetHDH="dataset/dataset_HDH.xlsx"

#json
label1_json_file= "dataset/label1/CNN_train_3c_relu.json"

c1_label2_json_file= "dataset/c1 label2/c1_label2.json"
c1_label3_cthdh_cht_json_file= "dataset/c1 label3/c1_cthdh_cht.json"
c1_label3_cthdh_kn_json_file= "dataset/c1 label3/c1_cthdh_kn.json"
c1_label3_cthdh_mt_json_file= "dataset/c1 label3/c1_cthdh_mt.json"
c1_label3_knhdh_json_file= "dataset/c1 label3/c1_knhdh.json"

c2_label2_ktmr_json_file= "dataset/c2 label2/c2_ktmr.json"
c2_label2_qlkg_json_file= "dataset/c2 label2/c2_qlkg.json"
c2_label2_tq_json_file= "dataset/c2 label2/c2_tq.json"
c2_label3_ktmr_kncht_json_file= "dataset/c2 label3/c2_ktmr_kncht.json"
c2_label3_qlkg_cht_json_file= "dataset/c2 label3/c2_qlkg_cht.json"
c2_label3_qlkg_kn_json_file= "dataset/c2 label3/c2_qlkg_kn.json"
c2_label3_qlkg_mt_json_file= "dataset/c2 label3/c2_qlkg_mt.json"
c2_label3_tq_cht_json_file= "dataset/c2 label3/c2_tq_cht.json"
c2_label3_tq_kn_json_file= "dataset/c2 label3/c2_tq_kn.json"
c2_label3_tq_mt_json_file= "dataset/c2 label3/c2_tq_mt.json"

c3_label2_dbtt_json_file= "dataset/c3 label2/c3_dbtt.json"
c3_label2_dptt_json_file= "dataset/c3 label2/c3_dptt.json"
c3_label2_kntt_json_file= "dataset/c3 label2/c3_kntt.json"
c3_label2_tttn_json_file= "dataset/c3 label2/c3_tttn.json"
c3_label3_dbtt_gtkn_json_file= "dataset/c3 label3/c3_dbtt_gtkn.json"
c3_label3_dptt_gtkn_json_file= "dataset/c3 label3/c3_dptt_gtkn.json"
c3_label3_kntt_gtkn_json_file= "dataset/c3 label3/c3_kntt_gtkn.json"
c3_label3_tttn_gtkn_json_file= "dataset/c3 label3/c3_tttn_gtkn.json"

c4_label2_bna_json_file= "dataset/c4 label2/c4_bna.json"
c4_label2_dc_json_file= "dataset/c4 label2/c4_dc.json"
c4_label2_pcbn_json_file= "dataset/c4 label2/c4_pcbn.json"
c4_label2_pdbn_json_file= "dataset/c4 label2/c4_pdbn.json"
c4_label2_ptbn_json_file= "dataset/c4 label2/c4_ptbn.json"
c4_label3_bna_ch_json_file= "dataset/c4 label3/c4_bna_ch.json"
c4_label3_bna_gt_json_file= "dataset/c4 label3/c4_bna_gt.json"
c4_label3_bna_kn_json_file= "dataset/c4 label3/c4_bna_kn.json"
c4_label3_dc_ch_json_file= "dataset/c4 label3/c4_dc_ch.json"
c4_label3_dc_gt_json_file= "dataset/c4 label3/c4_dc_gt.json"
c4_label3_dc_kn_json_file= "dataset/c4 label3/c4_dc_kn.json"
c4_label3_pcbn_gtch_json_file= "dataset/c4 label3/c4_pcbn_gtch.json"
c4_label3_pdbn_gtch_json_file= "dataset/c4 label3/c4_pdbn_gtch.json"
c4_label3_ptbn_gtch_json_file= "dataset/c4 label3/c4_ptbn_gtch.json"


#weight
label1_weight_file= "dataset/label1/label1.h5"

c1_label2_weight_file= "dataset/c1 label2/c1_label2-003-0.0558-1.0000.h5"
c1_label3_cthdh_cht_weight_file= "dataset/c1 label3/c1_cthdh_cht-001-0.3967-1.0000.h5"
c1_label3_cthdh_kn_weight_file= "dataset/c1 label3/c1_cthdh_kn-001-0.3683-1.0000.h5"
c1_label3_cthdh_mt_weight_file= "dataset/c1 label3/c1_cthdh_mt-001-0.4326-1.0000.h5"
c1_label3_knhdh_weight_file= "dataset/c1 label3/c1_knhdh-001-0.0172-1.0000.h5"

c2_label2_ktmr_weight_file= "dataset/c2 label2/c2_ktmr-013-0.0299-1.0000.h5"
c2_label2_qlkg_weight_file= "dataset/c2 label2/c2_qlkg-009-0.0282-1.0000.h5"
c2_label2_tq_weight_file= "dataset/c2 label2/c2_tq-005-0.0537-1.0000.h5"
c2_label3_ktmr_kncht_weight_file= "dataset/c2 label3/c2_ktmr_kn_cht-002-0.0494-1.0000.h5"
c2_label3_qlkg_cht_weight_file= "dataset/c2 label3/c2_qlkg_cht-002-0.1266-1.0000.h5"
c2_label3_qlkg_kn_weight_file= "dataset/c2 label3/c2_qlkg_kn-005-0.0481-1.0000.h5"
c2_label3_qlkg_mt_weight_file= "dataset/c2 label3/c2_qlkg_mt-004-0.0623-1.0000.h5"
c2_label3_tq_cht_weight_file= "dataset/c2 label3/c2_tq_cht-002-0.1257-1.0000.h5"
c2_label3_tq_kn_weight_file= "dataset/c2 label3/c2_tq_kn-004-0.0295-1.0000.h5"
c2_label3_tq_mt_weight_file= "dataset/c2 label3/c2_tq_mt-005-0.0490-1.0000.h5"

c3_label2_dbtt_weight_file= "dataset/c3 label2/c3_dbtt-001-0.4313-1.0000.h5"
c3_label2_dptt_weight_file= "dataset/c3 label2/c3_dptt-006-0.0289-1.0000.h5"
c3_label2_kntt_weight_file= "dataset/c3 label2/c3_kntt-005-0.0479-1.0000.h5"
c3_label2_tttn_weight_file= "dataset/c3 label2/c3_tttn-001-0.4356-1.0000.h5"
c3_label3_dbtt_gtkn_weight_file= "dataset/c3 label3/c3_dbtt_gt_kn-001-0.3574-1.0000.h5"
c3_label3_dptt_gtkn_weight_file= "dataset/c3 label3/c3_dptt_gt_kn-001-0.4500-1.0000.h5"
c3_label3_kntt_gtkn_weight_file= "dataset/c3 label3/c3_kntt_gt_kn-004-0.0570-1.0000.h5"
c3_label3_tttn_gtkn_weight_file= "dataset/c3 label3/c3_tttn_gt_kn-002-0.1583-1.0000.h5"


c4_label2_bna_weight_file= "dataset/c4 label2/c4_bna-003-0.0862-1.0000.h5"
c4_label2_dc_weight_file= "dataset/c4 label2/c4_dc-003-0.0364-1.0000.h5"
c4_label2_pcbn_weight_file= "dataset/c4 label2/c4_pcbn-008-0.0292-1.0000.h5"
c4_label2_pdbn_weight_file= "dataset/c4 label2/c4_pdbn-001-0.1260-0.8750.h5"
c4_label2_ptbn_weight_file= "dataset/c4 label2/c4_ptbn-001-0.0681-0.9375.h5"
c4_label3_bna_ch_weight_file= "dataset/c4 label3/c4_bna_ch-002-0.0745-1.0000.h5"
c4_label3_bna_gt_weight_file= "dataset/c4 label3/c4_bna_gt-001-0.3833-1.0000.h5"
c4_label3_bna_kn_weight_file= "dataset/c4 label3/c4_bna_kn-006-0.0427-1.0000.h5"
c4_label3_dc_ch_weight_file= "dataset/c4 label3/c4_dc_ch-003-0.0757-1.0000.h5"
c4_label3_dc_gt_weight_file= "dataset/c4 label3/c4_dc_gt-004-0.0477-1.0000.h5"
c4_label3_dc_kn_weight_file= "dataset/c4 label3/c4_dc_kn-001-0.4504-1.0000.h5"
c4_label3_pcbn_gtch_weight_file= "dataset/c4 label3/c4_pcbn_gtch-001-0.1032-1.0000.h5"
c4_label3_pdbn_gtch_weight_file= "dataset/c4 label3/c4_pdbn_gtch-003-0.0731-1.0000.h5"
c4_label3_ptbn_gtch_weight_file= "dataset/c4 label3/c4_ptbn_gtch-003-0.0604-1.0000.h5"

label1 = ['c1','c2','c3','c4']

c1_label2= ['CTHDH','KNHDH']
c1_label3_cthdh_cht= ['CHT','none']
c1_label3_cthdh_kn= ['KN','none']
c1_label3_cthdh_mt= ['MT','none']
c1_label3_knhdh= ['KN','MT']

c2_label2_ktmr= ['KTMR','none']
c2_label2_qlkg= ['QLKG','none']
c2_label2_tq= ['TQ','none']
c2_label3_ktmr_kncht= ['KN','cht']
c2_label3_qlkg_cht=['CHT','none']
c2_label3_qlkg_kn=['KN','none']
c2_label3_qlkg_mt=['MT','none']
c2_label3_tq_cht=['CHT','none']
c2_label3_tq_kn=['KN','none']
c2_label3_tq_mt=['MT','none']

c3_label2_dbtt= ['DB_TT','none']
c3_label2_dptt=['DP_TT','none']
c3_label2_kntt=['KN_TT','none']
c3_label2_tttn=['TT_TN','none']
c3_label3_dbtt_gtkn= ['GT','KN']
c3_label3_dptt_gtkn=['GT','KN']
c3_label3_kntt_gtkn=['GT','KN']
c3_label3_tttn_gtkn=['GT','KN']


c4_label2_bna=['BNA','none']
c4_label2_dc=['DC','none']
c4_label2_pcbn=['PCBN','none']
c4_label2_pdbn=['PDBN','none']
c4_label2_ptbn=['PTBN','none']
c4_label3_bna_ch=['CH','none']
c4_label3_bna_gt=['GT','none']
c4_label3_bna_kn=['KN','none']
c4_label3_dc_ch=['CH','none']
c4_label3_dc_gt=['GT','none']
c4_label3_dc_kn=['KN','none']
c4_label3_pcbn_gtch=['CH','GT']
c4_label3_pdbn_gtch=['CH','GT']
c4_label3_ptbn_gtch=['CH','GT']


EMBEDDING_DIM = 300
NUM_WORDS = 50000
max_length = 300
max_length1 = 30000
pad = ['post', 'pre']
test_num_full = 400

def load_ant_aspect_model(label_json_file, label_weight_file):
    json_file = open(label_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(label_weight_file)
    return model


def label_predict(str, name_label, tok_sam, sample_seq, model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    
    if len(str_temp)>1:
        sentences.append(str_temp)
        text = tok_sam.texts_to_sequences(sentences)
        # print("tok_sam")
        # print(text)
        seq = pad_sequences(text, maxlen=sample_seq.shape[1], padding='post')
        pred_ant = model.predict(seq) 
        temp_aspect_detect = name_label[np.argmax(pred_ant)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def load_full_data():
    train_data = pd.read_excel(url_datasetHDH, 'datasetHDH')

    val_data = train_data.sample(frac=0.12, random_state=42)  # cross validation: 1/10
    test_data = train_data.sample(frac=0.1, random_state=42)

    #test_data = train_data[34161:35463]
    # train_data = train_data.drop(val_data.index)
    # train_data = train_data.drop(test_data.index)

    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[/]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


tok_sam, sample_seq = load_full_data()
aspect_text= input()
#print("load file successful!")

label1_model = load_ant_aspect_model(label1_json_file, label1_weight_file)
label1_pred = label_predict(aspect_text, label1, tok_sam, sample_seq, label1_model)
#print("load file1 successful!")


#predict

def filler_data(l1,l2,l3):
  df = pd.read_excel(url_datasetHDH, 'datasetHDH')
  df1 = df[(df['label1']==''.join(l1)) & (df['label2']==''.join(l2))& (df['label3']==''.join(l3))] #viết hoa 
  df1.drop(['label1','label2','label3'], axis =1, inplace= True)

  #df1.to_excel('predicted.xlsx', sheet_name='predict', index = False)

  return df1


if label1_pred == ['c1']:
    c1_label2_model = load_ant_aspect_model(c1_label2_json_file, c1_label2_weight_file)
    c1_label2_pred = label_predict(aspect_text, c1_label2, tok_sam, sample_seq, c1_label2_model)

    

    if c1_label2_pred == ['CTHDH']:  
        c1_label3_cthdh_cht_model = load_ant_aspect_model(c1_label3_cthdh_cht_json_file, c1_label3_cthdh_cht_weight_file)
        c1_label3_cthdh_cht_pred = label_predict(aspect_text, c1_label3_cthdh_cht, tok_sam, sample_seq, c1_label3_cthdh_cht_model)

        if c1_label3_cthdh_cht_pred == ['CHT']:
          
            print("c1_cthdh_cht")
            
            
            # df1 = df[(df['label1']=='c1') & (df['label2']=='CTHDH')& (df['label3']=='CHT')] #viết hoa 
            # df1.drop(['id','label1','label2','label3'], axis =1, inplace= True)
            # df1.to_excel('c1_cthdh_cht.xlsx', sheet_name='predict', index = False) 
             
            df2= filler_data(label1_pred[0],c1_label2_pred[0],c1_label3_cthdh_cht_pred[0])

        else:
            
            c1_label3_cthdh_kn_model = load_ant_aspect_model(c1_label3_cthdh_kn_json_file, c1_label3_cthdh_kn_weight_file)
            c1_label3_cthdh_kn_pred = label_predict(aspect_text, c1_label3_cthdh_kn, tok_sam, sample_seq, c1_label3_cthdh_kn_model)
            
            if c1_label3_cthdh_kn_pred == ['KN']:
                print("c1_cthdh_kn")
                df2= filler_data(label1_pred[0],c1_label2_pred[0],c1_label3_cthdh_kn_pred[0])

            else:
                c1_label3_cthdh_mt_model = load_ant_aspect_model(c1_label3_cthdh_mt_json_file, c1_label3_cthdh_mt_weight_file)
                c1_label3_cthdh_mt_pred = label_predict(aspect_text, c1_label3_cthdh_mt, tok_sam, sample_seq, c1_label3_cthdh_mt_model)

                if c1_label3_cthdh_mt_pred == ['MT']:
                    print("c1_cthdh_mt")
                    df2=filler_data(label1_pred[0],c1_label2_pred[0],c1_label3_cthdh_mt_pred[0])
                else:
                    print('Không hợp lệ!')

    if c1_label2_pred == ['KNHDH']:
        c1_label3_knhdh_model = load_ant_aspect_model(c1_label3_knhdh_json_file, c1_label3_knhdh_weight_file)
        c1_label3_knhdh_pred = label_predict(aspect_text, c1_label3_knhdh, tok_sam, sample_seq, c1_label3_knhdh_model)

        if c1_label3_knhdh_pred == ['KN']:
            print("c1_knhdh_kn")
            df2=filler_data(label1_pred[0],c1_label2_pred[0],c1_label3_knhdh_pred[0])

        elif c1_label3_knhdh_pred == ['MT']:
            print("c1_knhdh_mt")  
            df2=filler_data(label1_pred[0],c1_label2_pred[0],c1_label3_knhdh_pred[0])

if label1_pred == ['c2']:
    c2_label2_ktmr_model = load_ant_aspect_model(c2_label2_ktmr_json_file, c2_label2_ktmr_weight_file)
    c2_label2_ktmr_pred = label_predict(aspect_text, c2_label2_ktmr, tok_sam, sample_seq, c2_label2_ktmr_model)
    
    c2_label2_tq_model = load_ant_aspect_model(c2_label2_tq_json_file, c2_label2_tq_weight_file)
    c2_label2_tq_pred = label_predict(aspect_text, c2_label2_tq, tok_sam, sample_seq, c2_label2_tq_model)

    if c2_label2_ktmr_pred == ['KTMR']:
        c2_label3_ktmr_kncht_model = load_ant_aspect_model(c2_label3_ktmr_kncht_json_file, c2_label3_ktmr_kncht_weight_file)
        c2_label3_ktmr_kncht_pred = label_predict(aspect_text, c2_label3_ktmr_kncht, tok_sam, sample_seq, c2_label3_ktmr_kncht_model)

        if c2_label3_ktmr_kncht_pred == ['KN']:
            print('c2_ktmr_kn')
            df2=filler_data(label1_pred[0],c2_label2_ktmr_pred[0],c2_label3_ktmr_kncht_pred[0])

        elif c2_label3_ktmr_kncht_pred == ['CHT']:
            print('c2_ktmr_cht')
            df2=filler_data(label1_pred[0],c2_label2_ktmr_pred[0],c2_label3_ktmr_kncht_pred[0])
    else:
        c2_label2_qlkg_model = load_ant_aspect_model(c2_label2_qlkg_json_file, c2_label2_qlkg_weight_file)
        c2_label2_qlkg_pred = label_predict(aspect_text, c2_label2_qlkg, tok_sam, sample_seq, c2_label2_qlkg_model)

    
        if c2_label2_qlkg_pred == ['QLKG']:
            c2_label3_qlkg_cht_model = load_ant_aspect_model(c2_label3_qlkg_cht_json_file, c2_label3_qlkg_cht_weight_file)
            c2_label3_qlkg_cht_pred = label_predict(aspect_text, c2_label2_ktmr, tok_sam, sample_seq, c2_label2_ktmr_model)
            
            

            if c2_label3_qlkg_cht_pred == ['CHT']:
                print('c2_qlkg_cht')
                df2=filler_data(label1_pred[0],c2_label2_qlkg_pred[0],c2_label3_qlkg_cht_pred[0])
            else :
                c2_label3_qlkg_kn_model = load_ant_aspect_model(c2_label3_qlkg_kn_json_file, c2_label3_qlkg_kn_weight_file)
                c2_label3_qlkg_kn_pred = label_predict(aspect_text, c2_label3_qlkg_kn, tok_sam, sample_seq, c2_label3_qlkg_kn_model)
            
                if c2_label3_qlkg_kn_pred == ['KN']:
                    print('c2_qlkg_kn')
                    df2=filler_data(label1_pred[0],c2_label2_qlkg_pred[0],c2_label3_qlkg_kn_pred[0])
                else :
                    c2_label3_qlkg_mt_model = load_ant_aspect_model(c2_label3_qlkg_mt_json_file, c2_label3_qlkg_mt_weight_file)
                    c2_label3_qlkg_mt_pred = label_predict(aspect_text, c2_label3_qlkg_mt, tok_sam, sample_seq, c2_label3_qlkg_mt_model)

                    if c2_label3_qlkg_mt_pred ==['MT']:
                        print('c2_qlkg_mt')
                        df2= filler_data(label1_pred[0],c2_label2_qlkg_pred[0],c2_label3_qlkg_mt_pred[0])
                    else :
                        print("Không hợp l")

        else :

            if c2_label2_tq_pred == ['TQ']:
                c2_label3_tq_cht_model = load_ant_aspect_model(c2_label3_tq_cht_json_file, c2_label3_tq_cht_weight_file)
                c2_label3_tq_cht_pred = label_predict(aspect_text, c2_label3_tq_cht, tok_sam, sample_seq, c2_label3_tq_cht_model)
                
                

                if c2_label3_tq_cht_pred == ['CHT']:
                    print('c2_tq_cht')
                    df2=filler_data(label1_pred[0],c2_label2_tq_pred[0],c2_label3_tq_cht_pred[0])
                  
                else :
                    c2_label3_tq_kn_model = load_ant_aspect_model(c2_label3_tq_kn_json_file, c2_label3_tq_kn_weight_file)
                    c2_label3_tq_kn_pred = label_predict(aspect_text, c2_label3_tq_kn, tok_sam, sample_seq, c2_label3_tq_kn_model)

                    if c2_label3_tq_kn_pred == ['KN']:
                        print('c2_tq_kn')
                        df2=filler_data(label1_pred[0],c2_label2_tq_pred[0],c2_label3_tq_kn_pred[0])
                    else :
                        c2_label3_tq_mt_model = load_ant_aspect_model(c2_label3_tq_mt_json_file, c2_label3_tq_mt_weight_file)
                        c2_label3_tq_mt_pred = label_predict(aspect_text, c2_label3_tq_mt, tok_sam, sample_seq, c2_label3_tq_mt_model)

                        if c2_label3_tq_mt_pred ==['MT']:
                            print('c2_tq_mt')
                            df2=filler_data(label1_pred[0],c2_label2_tq_pred[0],c2_label3_tq_mt_pred[0])
                        else :
                            print('Không hợp lệ!')

if label1_pred == ['c3']:
    c3_label2_dbtt_model = load_ant_aspect_model(c3_label2_dbtt_json_file, c3_label2_dbtt_weight_file)
    c3_label2_dbtt_pred = label_predict(aspect_text, c3_label2_dbtt, tok_sam, sample_seq, c3_label2_dbtt_model)
    
    
    

    if c3_label2_dbtt_pred == ['DB_TT']:
        c3_label3_dbtt_gtkn_model = load_ant_aspect_model(c3_label3_dbtt_gtkn_json_file, c3_label3_dbtt_gtkn_weight_file)
        c3_label3_dbtt_gtkn_pred = label_predict(aspect_text, c3_label3_dbtt_gtkn, tok_sam, sample_seq, c3_label3_dbtt_gtkn_model)
        

        if c3_label3_dbtt_gtkn_pred == ['GT']:
            print("c3_dbtt_gt")
            df2=filler_data(label1_pred[0],c3_label2_dbtt_pred[0],c3_label3_dbtt_gtkn_pred[0])

        elif c3_label3_dbtt_gtkn_pred == ['KN']:
            print("c3_dbtt_kn")
            df2=filler_data(label1_pred[0],c3_label2_dbtt_pred[0],c3_label3_dbtt_gtkn_pred[0])
    else:
        c3_label2_dptt_model = load_ant_aspect_model(c3_label2_dptt_json_file, c3_label2_dptt_weight_file)
        c3_label2_dptt_pred = label_predict(aspect_text, c3_label2_dptt, tok_sam, sample_seq, c3_label2_dptt_model)

        if c3_label2_dptt_pred == ['DP_TT']:
            c3_label3_dptt_gtkn_model = load_ant_aspect_model(c3_label3_dptt_gtkn_json_file, c3_label3_dptt_gtkn_weight_file)
            c3_label3_dptt_gtkn_pred = label_predict(aspect_text, c3_label3_dptt_gtkn, tok_sam, sample_seq, c3_label3_dptt_gtkn_model)

            if c3_label3_dptt_gtkn_pred == ['GT']:
                print("c3_dptt_gt")
                df2= filler_data(label1_pred[0],c3_label2_dptt_pred[0],c3_label3_dptt_gtkn_pred[0])

            elif c3_label3_dptt_gtkn_pred == ['KN']:
                print("c3_dptt_kn")
                df2=filler_data(label1_pred[0],c3_label2_dptt_pred[0],c3_label3_dptt_gtkn_pred[0])

        else :
            c3_label2_kntt_model = load_ant_aspect_model(c3_label2_kntt_json_file, c3_label2_kntt_weight_file)
            c3_label2_kntt_pred = label_predict(aspect_text, c3_label2_kntt, tok_sam, sample_seq, c3_label2_kntt_model)

            if c3_label2_kntt_pred == ['KN_TT']:
                c3_label3_kntt_gtkn_model = load_ant_aspect_model(c3_label3_kntt_gtkn_json_file, c3_label3_kntt_gtkn_weight_file)
                c3_label3_kntt_gtkn_pred = label_predict(aspect_text, c3_label3_kntt_gtkn, tok_sam, sample_seq, c3_label3_kntt_gtkn_model)

                if c3_label3_kntt_gtkn_pred == ['GT']:
                    print("c3_kntt_gt")
                    df2=filler_data(label1_pred[0],c3_label2_kntt_pred[0],c3_label3_kntt_gtkn_pred[0])

                else :
                    print("c3_kntt_kn")
                    df2=filler_data(label1_pred[0],c3_label2_kntt_pred[0],c3_label3_kntt_gtkn_pred[0])

            else :
                  c3_label2_tttn_model = load_ant_aspect_model(c3_label2_tttn_json_file, c3_label2_tttn_weight_file)
                  c3_label2_tttn_pred = label_predict(aspect_text, c3_label2_tttn, tok_sam, sample_seq, c3_label2_tttn_model)

                  if c3_label2_tttn_pred == ['TT_TN']:
                      c3_label3_tttn_gtkn_model = load_ant_aspect_model(c3_label3_tttn_gtkn_json_file, c3_label3_tttn_gtkn_weight_file)
                      c3_label3_tttn_gtkn_pred = label_predict(aspect_text, c3_label3_tttn_gtkn, tok_sam, sample_seq, c3_label3_tttn_gtkn_model)

                      if c3_label3_tttn_gtkn_pred == ['GT']:
                          print("c3_tttn_gt")
                          df2=filler_data(label1_pred[0],c3_label2_tttn_pred[0],c3_label3_tttn_gtkn_pred[0])

                      elif c3_label3_tttn_gtkn_pred == ['KN']:
                          print("c3_tttn_kn")
                          df2=filler_data(label1_pred[0],c3_label2_tttn_pred[0],c3_label3_tttn_gtkn_pred[0])

if label1_pred == ['c4']:
    c4_label2_bna_model = load_ant_aspect_model(c4_label2_bna_json_file, c4_label2_bna_weight_file)
    c4_label2_bna_pred = label_predict(aspect_text, c4_label2_bna, tok_sam, sample_seq, c4_label2_bna_model)
    
    
    
    

    if c4_label2_bna_pred == ['BNA']:
        c4_label3_bna_ch_model = load_ant_aspect_model(c4_label3_bna_ch_json_file, c4_label3_bna_ch_weight_file)
        c4_label3_bna_ch_pred = label_predict(aspect_text, c4_label3_bna_ch, tok_sam, sample_seq, c4_label3_bna_ch_model)
        
        
        

        if c4_label3_bna_ch_pred == ['CH']:
            print("c4_bna_ch")
            df2=filler_data(label1_pred[0],c4_label2_bna_pred[0],c4_label3_bna_ch_pred[0])

        else :
            c4_label3_bna_gt_model = load_ant_aspect_model(c4_label3_bna_gt_json_file, c4_label3_bna_gt_weight_file)
            c4_label3_bna_gt_pred = label_predict(aspect_text, c4_label3_bna_gt, tok_sam, sample_seq, c4_label3_bna_gt_model)

            if c4_label3_bna_gt_pred== ['GT']:
                print("c4_bna_gt")
                df2=filler_data(label1_pred[0],c4_label2_bna_pred[0],c4_label3_bna_gt_pred[0])

            else :
                c4_label3_bna_kn_model = load_ant_aspect_model(c4_label3_bna_kn_json_file, c4_label3_bna_kn_weight_file)
                c4_label3_bna_kn_pred = label_predict(aspect_text, c4_label3_bna_kn, tok_sam, sample_seq, c4_label3_bna_kn_model)

                if c4_label3_bna_kn_pred == ['KN']:
                    print("c4_bna_kn")
                    df2=filler_data(label1_pred[0],c4_label2_bna_pred[0],c4_label3_bna_kn_pred[0])
                else :
                    print('Không hợp lệ!')
    
    else :
        c4_label2_dc_model = load_ant_aspect_model(c4_label2_dc_json_file, c4_label2_dc_weight_file)
        c4_label2_dc_pred = label_predict(aspect_text, c4_label2_dc, tok_sam, sample_seq, c4_label2_dc_model)

        if c4_label2_dc_pred == ['DC']:
            c4_label3_dc_ch_model = load_ant_aspect_model(c4_label3_dc_ch_json_file, c4_label3_dc_ch_weight_file)
            c4_label3_dc_ch_pred = label_predict(aspect_text, c4_label3_dc_ch, tok_sam, sample_seq, c4_label3_dc_ch_model)
            
            

            if c4_label3_dc_ch_pred == ['CH']:
                print("c4_dc_ch")
                df2=filler_data(label1_pred[0],c4_label2_dc_pred[0],c4_label3_dc_ch_pred[0])
            else :
                c4_label3_dc_gt_model = load_ant_aspect_model(c4_label3_dc_gt_json_file, c4_label3_dc_gt_weight_file)
                c4_label3_dc_gt_pred = label_predict(aspect_text, c4_label3_dc_gt, tok_sam, sample_seq, c4_label3_dc_gt_model)

                if c4_label3_dc_gt_pred== ['GT']:
                    print("c4_dc_gt")
                    df2=filler_data(label1_pred[0],c4_label2_dc_pred[0],c4_label3_dc_gt_pred[0])
                else :
                    c4_label3_dc_kn_model = load_ant_aspect_model(c4_label3_dc_kn_json_file, c4_label3_dc_kn_weight_file)
                    c4_label3_dc_kn_pred = label_predict(aspect_text, c4_label3_dc_kn, tok_sam, sample_seq, c4_label3_dc_kn_model)

                    if c4_label3_dc_kn_pred == ['KN']:
                        print("c4_dc_kn")
                        df2=filler_data(label1_pred[0],c4_label2_dc_pred[0],c4_label3_dc_kn_pred[0])
                    else :
                        print("Không hợp lệ!")

        else :
            c4_label2_pcbn_model = load_ant_aspect_model(c4_label2_pcbn_json_file, c4_label2_pcbn_weight_file)
            c4_label2_pcbn_pred = label_predict(aspect_text, c4_label2_pcbn, tok_sam, sample_seq, c4_label2_pcbn_model)

            if c4_label2_pcbn_pred== ['PCBN']:
                c4_label3_pcbn_gtch_model = load_ant_aspect_model(c4_label3_pcbn_gtch_json_file, c4_label3_pcbn_gtch_weight_file)
                c4_label3_pcbn_gtch_pred = label_predict(aspect_text, c4_label3_pcbn_gtch, tok_sam, sample_seq, c4_label3_pcbn_gtch_model)

                if c4_label3_pcbn_gtch_pred == ['CH']:
                    print("c4_pcbn_ch")
                    df2=filler_data(label1_pred[0],c4_label2_pcbn_pred[0],c4_label3_pcbn_gtch_pred[0])
                
                elif c4_label3_pcbn_gtch_pred== ['GT']:
                    print("c4_pcbn_gt")
                    df2=filler_data(label1_pred[0],c4_label2_pcbn_pred[0],c4_label3_pcbn_gtch_pred[0])
            
            else :
                c4_label2_pdbn_model = load_ant_aspect_model(c4_label2_pdbn_json_file, c4_label2_pdbn_weight_file)
                c4_label2_pdbn_pred = label_predict(aspect_text, c4_label2_pdbn, tok_sam, sample_seq, c4_label2_pdbn_model)

                if c4_label2_pdbn_pred == ['PDBN']:
                    c4_label3_pdbn_gtch_model = load_ant_aspect_model(c4_label3_pdbn_gtch_json_file, c4_label3_pdbn_gtch_weight_file)
                    c4_label3_pdbn_gtch_pred = label_predict(aspect_text, c4_label3_pdbn_gtch, tok_sam, sample_seq, c4_label3_pdbn_gtch_model)

                    if c4_label3_pdbn_gtch_pred == ['CH']:
                        print("c4_pdbn_ch")
                        df2=filler_data(label1_pred[0],c4_label2_pdbn_pred[0],c4_label3_pdbn_gtch_pred[0])
                    
                    elif c4_label3_pdbn_gtch_pred== ['GT']:
                        print("c4_pdbn_gt")
                        df2=filler_data(label1_pred[0],c4_label2_pdbn_pred[0],c4_label3_pdbn_gtch_pred[0])

                else :
                    c4_label2_ptbn_model = load_ant_aspect_model(c4_label2_ptbn_json_file, c4_label2_ptbn_weight_file)
                    c4_label2_ptbn_pred = label_predict(aspect_text, c4_label2_ptbn, tok_sam, sample_seq, c4_label2_ptbn_model)

                if c4_label2_ptbn_pred== ['PTBN']:
                    c4_label3_ptbn_gtch_model = load_ant_aspect_model(c4_label3_ptbn_gtch_json_file, c4_label3_ptbn_gtch_weight_file)
                    c4_label3_ptbn_gtch_pred = label_predict(aspect_text, c4_label3_ptbn_gtch, tok_sam, sample_seq, c4_label3_ptbn_gtch_model)
                    
                    if c4_label3_ptbn_gtch_pred == ['CH']:
                        print("c4_ptbn_ch")
                        df2=filler_data(label1_pred[0],c4_label2_ptbn_pred[0],c4_label3_ptbn_gtch_pred[0])
                    
                    elif c4_label3_ptbn_gtch_pred== ['GT']:
                        print("c4_ptbn_gt")
                        df2=filler_data(label1_pred[0],c4_label2_ptbn_pred[0],c4_label3_ptbn_gtch_pred[0])



# TF-IDF and Cosine similarity
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import euclidean_distances


train_data = df2['text'].to_list()
doc= [aspect_text]

cv=CountVectorizer()
word_count_vector=cv.fit_transform(train_data)
# print(word_count_vector.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(word_count_vector)
# print('train')
# print(X_train_tfidf)


X_new_counts = cv.transform(doc)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# print('input')
# print(X_new_tfidf)
'''

'''
# dùng cosine similarity để tính khoảng cách giữa 2 vector
cosin= cosine_similarity(X_new_counts,X_train_tfidf)
#print(cosin)

# sắp xếp theo thứ tự tăng dần nhưng vẫn giữ vị trí index
sorted_rank = np.argsort(cosin)
#print(sorted_rank)

sorted= []
for i in sorted_rank:
  sorted=i
#print(sorted)



print("Câu hỏi của bạn: ",aspect_text)
print('--------------------')
print("Câu trả lời: \n",train_data[sorted[len(sorted) - 1]]);
print('--------------------')
print("Câu trả lời liên quan: \n1.",train_data[sorted[len(sorted) - 2]]);
print("2.",train_data[sorted[len(sorted) - 3]]);
print("3. ",train_data[sorted[len(sorted) - 4]]);

print(' %s seconds' % (time.time() - start))

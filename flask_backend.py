#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage
from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from constants import STORED_MODEL_PATH

app = Flask(__name__)

# Maps original dataset names to tranform the received packets 
RENAME_MAP = {
    # 1. Protocol
    'protocol':                  'Protocol',
    # 2. Flow Duration
    'flow_duration':             'Flow Duration',
    # 3–6. Rates
    'flow_byts_s':               'Flow Bytes/s',
    'flow_pkts_s':               'Flow Packets/s',
    'fwd_pkts_s':                'Fwd Packets/s',
    'bwd_pkts_s':                'Bwd Packets/s',
    # 7–10. Packet counts & totals
    'tot_fwd_pkts':              'Total Fwd Packets',
    'tot_bwd_pkts':              'Total Backward Packets',
    'totlen_fwd_pkts':           'Fwd Packets Length Total',
    'totlen_bwd_pkts':           'Bwd Packets Length Total',
    # 11–18. Packet length stats forward/backward
    'fwd_pkt_len_max':           'Fwd Packet Length Max',
    'fwd_pkt_len_min':           'Fwd Packet Length Min',
    'fwd_pkt_len_mean':          'Fwd Packet Length Mean',
    'fwd_pkt_len_std':           'Fwd Packet Length Std',
    'bwd_pkt_len_max':           'Bwd Packet Length Max',
    'bwd_pkt_len_min':           'Bwd Packet Length Min',
    'bwd_pkt_len_mean':          'Bwd Packet Length Mean',
    'bwd_pkt_len_std':           'Bwd Packet Length Std',
    # 19–22. Overall packet length
    'pkt_len_max':               'Packet Length Max',
    'pkt_len_min':               'Packet Length Min',
    'pkt_len_mean':              'Packet Length Mean',
    'pkt_len_std':               'Packet Length Std',
    'pkt_len_var':               'Packet Length Variance',
    # 23–24. Header lengths
    'fwd_header_len':            'Fwd Header Length',
    'bwd_header_len':            'Bwd Header Length',
    # 25–26. Forward segment and data packets
    'fwd_seg_size_min':          'Fwd Seg Size Min',
    'fwd_act_data_pkts':         'Fwd Act Data Packets',
    # 27–30. Flow IAT stats
    'flow_iat_mean':             'Flow IAT Mean',
    'flow_iat_std':              'Flow IAT Std',
    'flow_iat_max':              'Flow IAT Max',
    'flow_iat_min':              'Flow IAT Min',
    # 31–35. Fwd IAT stats
    'fwd_iat_tot':               'Fwd IAT Total',
    'fwd_iat_mean':              'Fwd IAT Mean',
    'fwd_iat_std':               'Fwd IAT Std',
    'fwd_iat_max':               'Fwd IAT Max',
    'fwd_iat_min':               'Fwd IAT Min',
    # 36–40. Bwd IAT stats
    'bwd_iat_tot':               'Bwd IAT Total',
    'bwd_iat_mean':              'Bwd IAT Mean',
    'bwd_iat_std':               'Bwd IAT Std',
    'bwd_iat_max':               'Bwd IAT Max',
    'bwd_iat_min':               'Bwd IAT Min',
    # 41–44. TCP PSH/URG flags
    'fwd_psh_flags':             'Fwd PSH Flags',
    'bwd_psh_flags':             'Bwd PSH Flags',
    'fwd_urg_flags':             'Fwd URG Flags',
    'bwd_urg_flags':             'Bwd URG Flags',
    # 45–52. TCP flag counts
    'fin_flag_cnt':              'FIN Flag Count',
    'syn_flag_cnt':              'SYN Flag Count',
    'rst_flag_cnt':              'RST Flag Count',
    'psh_flag_cnt':              'PSH Flag Count',
    'ack_flag_cnt':              'ACK Flag Count',
    'urg_flag_cnt':              'URG Flag Count',
    'ece_flag_cnt':              'ECE Flag Count',
    'cwe_flag_count':            'CWE Flag Count',
    # 53. Down/Up Ratio
    'down_up_ratio':             'Down/Up Ratio',
    # 54. Avg Packet Size
    'pkt_size_avg':              'Avg Packet Size',
    # 55–56. Initial window sizes
    'init_fwd_win_byts':         'Init Fwd Win Bytes',
    'init_bwd_win_byts':         'Init Bwd Win Bytes',
    # 57–60. Active/Idle stats
    'active_max':                'Active Max',
    'active_min':                'Active Min',
    'active_mean':               'Active Mean',
    'active_std':                'Active Std',
    'idle_max':                  'Idle Max',
    'idle_min':                  'Idle Min',
    'idle_mean':                 'Idle Mean',
    'idle_std':                  'Idle Std',
    # 61–64. Bulk rates (bytes/packets)
    'fwd_byts_b_avg':            'Fwd Avg Bytes/Bulk',
    'fwd_pkts_b_avg':            'Fwd Avg Packets/Bulk',
    'bwd_byts_b_avg':            'Bwd Avg Bytes/Bulk',
    'bwd_pkts_b_avg':            'Bwd Avg Packets/Bulk',
    # 65–66. Bulk rates (rate)
    'fwd_blk_rate_avg':          'Fwd Avg Bulk Rate',
    'bwd_blk_rate_avg':          'Bwd Avg Bulk Rate',
    # 67–68. Segment size averages
    'fwd_seg_size_avg':          'Avg Fwd Segment Size',
    'bwd_seg_size_avg':          'Avg Bwd Segment Size',
    # 69–72. Subflow counts & bytes
    'subflow_fwd_pkts':          'Subflow Fwd Packets',
    'subflow_bwd_pkts':          'Subflow Bwd Packets',
    'subflow_fwd_byts':          'Subflow Fwd Bytes',
    'subflow_bwd_byts':          'Subflow Bwd Bytes',
}

# Maps original dataset typing to tranform the received packets 
TYPE_MAP = {
    'Protocol':                  'int8',
    'Flow Duration':             'int32',
    'Total Fwd Packets':         'int32',
    'Total Backward Packets':    'int32',
    'Fwd Packets Length Total':  'int32',
    'Bwd Packets Length Total':  'int32',
    'Fwd Packet Length Max':     'int16',
    'Fwd Packet Length Min':     'int16',
    'Fwd Packet Length Mean':    'float32',
    'Fwd Packet Length Std':     'float32',
    'Bwd Packet Length Max':     'int16',
    'Bwd Packet Length Min':     'int16',
    'Bwd Packet Length Mean':    'float32',
    'Bwd Packet Length Std':     'float32',
    'Flow Bytes/s':              'float64',
    'Flow Packets/s':            'float64',
    'Flow IAT Mean':             'float32',
    'Flow IAT Std':              'float32',
    'Flow IAT Max':              'int32',
    'Flow IAT Min':              'int32',
    'Fwd IAT Total':             'int32',
    'Fwd IAT Mean':              'float32',
    'Fwd IAT Std':               'float32',
    'Fwd IAT Max':               'int32',
    'Fwd IAT Min':               'int32',
    'Bwd IAT Total':             'int32',
    'Bwd IAT Mean':              'float32',
    'Bwd IAT Std':               'float32',
    'Bwd IAT Max':               'int32',
    'Bwd IAT Min':               'int32',
    'Fwd PSH Flags':             'int8',
    'Bwd PSH Flags':             'int8',
    'Fwd URG Flags':             'int8',
    'Bwd URG Flags':             'int8',
    'Fwd Header Length':         'int64',
    'Bwd Header Length':         'int32',
    'Fwd Packets/s':             'float32',
    'Bwd Packets/s':             'float32',
    'Packet Length Min':         'int16',
    'Packet Length Max':         'int16',
    'Packet Length Mean':        'float32',
    'Packet Length Std':         'float32',
    'Packet Length Variance':    'float32',
    'FIN Flag Count':            'int8',
    'SYN Flag Count':            'int8',
    'RST Flag Count':            'int8',
    'PSH Flag Count':            'int8',
    'ACK Flag Count':            'int8',
    'URG Flag Count':            'int8',
    'CWE Flag Count':            'int8',
    'ECE Flag Count':            'int8',
    'Down/Up Ratio':             'int16',
    'Avg Packet Size':           'float32',
    'Avg Fwd Segment Size':      'float32',
    'Avg Bwd Segment Size':      'float32',
    'Fwd Avg Bytes/Bulk':        'int8',
    'Fwd Avg Packets/Bulk':      'int8',
    'Fwd Avg Bulk Rate':         'int8',
    'Bwd Avg Bytes/Bulk':        'int8',
    'Bwd Avg Packets/Bulk':      'int8',
    'Bwd Avg Bulk Rate':         'int8',
    'Subflow Fwd Packets':       'int32',
    'Subflow Fwd Bytes':         'int32',
    'Subflow Bwd Packets':       'int32',
    'Subflow Bwd Bytes':         'int32',
    'Init Fwd Win Bytes':        'int32',
    'Init Bwd Win Bytes':        'int32',
    'Fwd Act Data Packets':      'int32',
    'Fwd Seg Size Min':          'int32',
    'Active Mean':               'float32',
    'Active Std':                'float32',
    'Active Max':                'int32',
    'Active Min':                'int32',
    'Idle Mean':                 'float32',
    'Idle Std':                  'float32',
    'Idle Max':                  'int32',
    'Idle Min':                  'int32',
}

model = tf.keras.models.load_model(STORED_MODEL_PATH)

@app.route('/send_traffic', methods=['POST'])
def receive_traffic():
    data = None
    try:
        data = request.get_json(force=True)
        
        df = pd.json_normalize(data, record_path='flows')
        meta = ['src_ip','dst_ip','src_port','dst_port','src_mac','dst_mac','timestamp']
        
        df = df.drop(columns=meta)
        df_renamed = df.rename(columns=RENAME_MAP)
        df_typed = df_renamed.astype(TYPE_MAP)
        
        expected_features_ordered = list(TYPE_MAP.keys())
        df_ordered = df_typed[expected_features_ordered]
        
        # Make predictions
        predictions = model.predict(df_ordered, batch_size=32)   
            
        print("[PREDICTIONS]")
        print(predictions)
        
        return jsonify({
            "status": "ok",
            "predictions": predictions.tolist()
        }), 200
            
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
        

    return jsonify({"status": "ok", "received": data}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

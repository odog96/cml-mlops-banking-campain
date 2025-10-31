import numpy as np
import json, requests
import pandas as pd
import random
import cmlapi
import cdsw
import os
import time
from sklearn.metrics import root_mean_squared_error

# this will be run as a job 

if os.environ.get("MODEL_NAME") == "":
    os.environ["MODEL_NAME"] = "LSTM-2"
if os.environ.get("PROJECT_NAME") == "":
    os.environ["PROJECT_NAME"] = "SDWAN"


# Ensure that Model Name matches variable above

window = 24

# function(s)
def dataframe_to_json(df):
    """Create a dictionary that will later be converted to a JSON object
    """
    data = {
        'rx_gbs': df['rx_gbs'].iloc[-(window+1):].tolist(),
        'tx_gbs': df['tx_gbs'].iloc[-(window+1):].tolist(),
        'time': [df['time'].iloc[-1]],  # Only the last time entry
        'link': [df['link'].iloc[-1]],  # Only the last link entry
        'location': [df['location'].iloc[-1]]  # Only the last location entry
    }
    
    # build embedded dictionary step 1
    request_dict = {"request":data}

    # access key will be end point specific
    BackDict = {"accessKey":model_key}
    BackDict.update(request_dict)
    request_dict=BackDict
    
    return request_dict

# read data frame (or table data)
df = pd.read_csv('data/simple_synthetic_data.csv')

# sort data by time 
df.sort_values(by=['time'],inplace=True)


##########################################################################
### CML Model API 
# model parameter configuration
# there are series of steps required to cr_number, which is required to configure the model 
# endpoint payload correctly - i.e., make sure we're calling the right model

model_name = "LSTM-2"
#os.environ["MODEL_NAME"]
project_name = "SDWAN"
#os.environ["PROJECT_NAME"]
client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
target_model = client.list_all_models(search_filter=json.dumps({"name": model_name}))
model_key = target_model.models[0].access_key
model_url = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model")

# lines below get at the most recent deployment of the model to get deployment crn
proj_id = client.list_projects(search_filter=json.dumps({"name":project_name })).projects[0].id
mod_id = target_model.models[0].id
build_list = client.list_model_builds(project_id = proj_id, model_id = mod_id,sort='created_at')
build_id = build_list.model_builds[-1].id
model_deployments = client.list_model_deployments(project_id=proj_id,model_id=mod_id,build_id=build_id)
cr_number = model_deployments.model_deployments[0].crn
##########################################################################


# forecasting selection choices. 

response_labels_sample = []

stride = 12        # determines frequecy of model inference
                   # how many observations between forecast requests
load_frequency = 2 # determines how often ground truth is loaded 
                   # ratio of number of observations between load jobs
                   # number of observations between forecast requests
                   # e.g., ground truth job every 24 hours, forecast request every 12, therefore load_frequency is 2 
#load_lag = 1       # number of observation sets to load each load
                   # this will determine heap size
    
m_window = 14      # Monitoring window.  Currently setting to weekly. 7 x 2 times/ day

horizon = 6

# code below slides through new data
# each time taking window then taking 
# 1 step forward of size 'stride'


for j,i in enumerate(range(0,df.shape[0]-(window),stride)):
    temp_df = df.iloc[i:window+1+i,:] # 
    # convert input to json
    request_dict = dataframe_to_json(temp_df)
    
    response = requests.post(model_url, data=json.dumps(request_dict), headers={'Content-Type': 'application/json'})
    #time.sleep(1)
    
    new_item = {
        "uuid": response.json()["response"]["uuid"],
        "rx_bytes_forecast": response.json()['response']['prediction']['rx_bytes'],
        "tx_bytes_forecast": response.json()['response']['prediction']['tx_bytes'],
        "horizon_time": response.json()['response']['prediction']['time'],
        "timestamp_ms": int(round(time.time() * 1000)),
    }
    response_labels_sample.append(new_item)
    
# Load Ground Truth phase
    if (j + 1) % load_frequency == 0:
        load_flag = True
        offset = load_frequency
    elif j == len(range(0, df.shape[0] - (horizon) - (window), stride)) - 1:
        load_flag = True
        offset = (j + 1) % load_frequency
    else:
        load_flag = False
        
    if load_flag:
        recent_items = response_labels_sample[-offset:]  # Get last load_frequency elements

        for val in recent_items:
            # Extract the horizon times for this entry
            horizon_times = val['horizon_time']

            # Filter the DataFrame rows where the 'time' column matches any of the horizon times
            filtered_horizon_times = [t.replace('T', ' ') for t in horizon_times]
            matched_rows = df[df['time'].isin(filtered_horizon_times)]
            
            # Extract 'rx_gbs' and 'tx_gbs' values for these rows
            rx_gbs_values = matched_rows['rx_gbs'].tolist()
            tx_gbs_values = matched_rows['tx_gbs'].tolist()
            
            cdsw.track_delayed_metrics({"final_rx_gbs_label": rx_gbs_values,"final_tx_gbs_label": tx_gbs_values}, val["uuid"])
            
    # monitoring step - Calculates desired model metrics based on predifined 
    # frequency
        # Monitoring step
    if (j + 1) % m_window == 0:
        monitor_flag = True
        m_offset = m_window
    elif j == len(range(0, df.shape[0] - (horizon) - (window), stride)) - 1:
        monitor_flag = True
        m_offset = (j + 1) % m_window
    else:
        monitor_flag = False
    
    if monitor_flag:
        m_recent_items = response_labels_sample[-m_offset:]  # Get last m_window elements

        print('length of m recent items', len(m_recent_items))
        print('iteration',j)
        
        start_timestamp_ms = m_recent_items[0]["timestamp_ms"]
        end_timestamp_ms = m_recent_items[-1]["timestamp_ms"]
        
        # Aggregate all timestamps and predictions from recent items
        all_horizon_times = []
        all_rx_predictions = []
        all_tx_predictions = []

        for item in m_recent_items:
            all_horizon_times.extend(item['horizon_time'])
            all_rx_predictions.extend(item['rx_bytes_forecast'])
            all_tx_predictions.extend(item['tx_bytes_forecast'])
            
        # Filter rows that match any of the collected timestamps
        filtered_all_horizon_times = [t.replace('T', ' ') for t in all_horizon_times]
        matched_rows = df[df['time'].isin(filtered_all_horizon_times)]
        
        # Get actual 'rx_gbs' and 'tx_gbs' values
        actual_rx = matched_rows['rx_gbs'].values #.tolist()
        actual_tx = matched_rows['tx_gbs'].values #.tolist()
        
        # convert predictions to np array
        all_rx_predictions = np.array(all_rx_predictions)
        all_tx_predictions = np.array(all_tx_predictions)
        
        # simplified RMSE calc ' please update as required'
        rmse_rx = root_mean_squared_error(actual_rx,all_rx_predictions)
        rmse_tx = root_mean_squared_error(actual_tx,all_tx_predictions)

        # update this to include an aggregate score - look to see if we can do 2 d score calc.
        
        rmse = (rmse_rx + rmse_tx)/2

        cdsw.track_aggregate_metrics(
                {"rmse": rmse},
                start_timestamp_ms,
                end_timestamp_ms,
                model_deployment_crn=cr_number,
            )
        
       
        # kick off check model job.. have to to wait until at least 2nds week. 

        # Get the identifier of the current project
        
        job_name = 'check_model'
        target_job = client.list_jobs(proj_id, search_filter=json.dumps({"name": job_name}))
        
        if (j+1) > m_window:
            go = True
            while go:
                # error handling if prior job is still running
                try:
                    job_run = client.create_job_run(cmlapi.CreateJobRunRequest(),project_id = proj_id, job_id = target_job.jobs[0].id)
                    go = False
                except:
                     time.sleep(5)
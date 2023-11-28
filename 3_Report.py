from Home import st
from Home import face_rec
import pandas as pd

st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

# Retrieve logs data and show in Report.py
# extract data from redis list
name = 'attendance:logs'

def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)  # extract all data from the redis database
    return logs_list

# Function to convert logs to DataFrame
def logs_to_dataframe(logs_list):
    # Assuming logs are in the format "Name@Role@Timestamp"
    logs_data = [log.decode('utf-8').split('@') for log in logs_list]
    columns = ['Name', 'Role', 'Timestamp']
    df = pd.DataFrame(logs_data, columns=columns)
    return df


# tabs to show the info
tab1, tab2 = st.tabs(['Registered Data', 'Logs'])

with tab1:
    if st.button('Refresh Data'):
        # Retrieve the data from Redis Database
        with st.spinner('Retrieving Data from Redis DB ...'):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('Refresh Logs'):
        logs_list = load_logs(name=name)
        logs_df = logs_to_dataframe(logs_list)
        st.dataframe(logs_df)

import argilla as rg


hf_space_url = "https://alikhan0100u-crs-rlhf-user-study.hf.space"
key = "ali0100u.apikey"

rg.init(
    api_url=hf_space_url,
    api_key=key,
    )


# Assume we distribute the workload in one dataset across multiple labelers
feedback = rg.FeedbackDataset.from_argilla(
    name="user-study-data-3",
    workspace="ws-run-3"
)

for i,record in enumerate(feedback.records):
    if record.responses is None or len(record.responses) == 0:
        continue
    print(f'==reading response for record {i+1}')


    for response in record.responses:
        uname = rg.User.from_id(response.user_id)
        if response.status == "submitted":
            rank_data = {v.value:v.rank for v in response.values["response_ranking"].value}
            print(f'user:{uname.username} rank:{rank_data}')

# # Define an empty list to store the triplets
# triplets = []

# # Loop over all records in the dataset
# for record in feedback.records:
#     # Ensure that the record has responses
#     if record.responses is None or len(record.responses) == 0:
#         print('record not completed')
#         continue

#     # Ensure the response has been submitted (not discarded)
#     response = record.responses[0]

#     if response.status == 'submitted':
#         # Get the ranking value from the response for the preferred and least preferred
#         # responses, assuming there are no ties
#         rank = [v["value"] for v in response.values["response_ranking"]]


# # Now, "triplets" is a list of dictionaries, each containing a prompt and the associated
# # preferred and less preferred responses
        

# from argilla import rg
# user = rg.User.create("my-user1","use-12345678",first_name='Luke',role='annotator') # or `User.from_id("...")`
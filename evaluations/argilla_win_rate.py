import argilla as rg

hf_space_url = "https://alikhan0100u-crs-rlhf-user-study.hf.space"
key = "ali0100u.apikey"

rg.init(
    api_url=hf_space_url,
    api_key=key,
    )

# Assume we distribute the workload in one dataset across multiple labelers
for run in range(3):
    feedback = rg.FeedbackDataset.from_argilla(
        workspace = f'ws-run-{run+1}',
        name=f"user-study-data-{run+1}",

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

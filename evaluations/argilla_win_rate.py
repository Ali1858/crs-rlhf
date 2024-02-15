import argilla as rg
from evaluations.tideman_ranking import ranked_pairs
import json

num_run = 3
hf_space_url = "https://alikhan0100u-crs-rlhf-user-study.hf.space"
key = "ali0100u.apikey"

rg.init(
    api_url=hf_space_url,
    api_key=key,
    )

def get_key_and_dataset():
    all_argilla_dataset = {}
    with open('evaluations/argilla_data/all_keys.json','r') as f:
        all_keys = json.load(f)
    
    for run in range(num_run):
        with open(f'evaluations/argilla_data/humaneval_p_09_t_08_run_{run}.json','r') as f:
            all_argilla_dataset[f'run_num{run}'] = json.load(f)
    return all_keys, all_argilla_dataset


def get_feedbacks(all_keys,all_argilla_dataset,run):


    ws_name = f'ws-run-{run+1}'
    # get feedback dataset for each workspace
    feedback = rg.FeedbackDataset.from_argilla(
        workspace=ws_name,
        name=f"user-study-data-{run+1}",
    )
    
    prompt_mapping = all_keys[f'run_num{run}']
    argilla_dataset = all_argilla_dataset[f'run_num{run}']
    feedback_records = feedback.records
    all_prompt_ranking = []
    all_prompt_ranking_original = []
    user_progress = {}
    
    for i,(feedback_record,prompt,argilla_data)in enumerate(zip(feedback_records,prompt_mapping.keys(),argilla_dataset)):
        # perform sanity check
        assert feedback_record.fields["instruction"] == prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[0]
        assert feedback_record.fields["response-1"] == argilla_data["response-1"]
        assert feedback_record.fields["response-2"] == argilla_data["response-2"]
        assert feedback_record.fields["response-3"] == argilla_data["response-3"]


        mapping = prompt_mapping[prompt]
        if feedback_record.responses is None or len(feedback_record.responses) == 0:
            continue

        each_prompt_votings = []
        each_voting_original = []
        for response in feedback_record.responses:
            uname = rg.User.from_id(response.user_id)
            if response.status == "submitted":# and uname.username == "bhup8":
                # Prepare dict to save original feedback as a backup
                original_feedback_dict = {v.value:v.rank for v in response.values["response_ranking"].value}
                original_feedback_dict['name'] = uname.username
                original_feedback_dict["user_id"] = str(response.user_id)
                original_feedback_dict["inserted_at"] = str(response.inserted_at)

                each_voting_original.append(original_feedback_dict)
                if uname.username in user_progress:
                    user_progress[uname.username] += 1
                else:
                    user_progress[uname.username] = 1

                rank_data = {
                    mapping[
                        str(
                            int(v.value.split('-')[-1])-1
                            )
                            ] : v.rank 
                            for v in response.values["response_ranking"].value
                            }
                # sort model name according to rank
                sorted_rank_data = dict(sorted(rank_data.items(), key=lambda item: item[1]))
                # get model names
                rank_model = list(sorted_rank_data.keys())
                each_prompt_votings.append(rank_model)

        all_prompt_ranking_original.append(each_voting_original)

        if len(each_prompt_votings) > 1:
            # merge all voitings using tideman voting method 
            all_prompt_ranking.append(ranked_pairs(each_prompt_votings))
        elif len(each_prompt_votings) > 0:
            all_prompt_ranking.append(each_prompt_votings[0])

    if len(all_prompt_ranking_original) > 0:
        with open(f'evaluations/argilla_data/original-user-study-{ws_name}.json','w') as f:
            json.dump(all_prompt_ranking_original, f, ensure_ascii=False, indent=4)
    print(f'Progress of each users out of 20 promts :{user_progress}')
    return all_prompt_ranking


for run in range(num_run):
    ws_name = f'ws-run-{run+1}'
    print(f'=== Getting feedback dataset for workshop {ws_name} =======.')

    all_keys, all_argilla_dataset = get_key_and_dataset()
    all_prompt_ranking = get_feedbacks(all_keys,all_argilla_dataset,run)

    # calculate final rank
    points_for_rank = {0: 3/3, 1: 2/3, 2: 1/3}
    if len(all_prompt_ranking) > 0:
        
        with open(f'evaluations/argilla_data/user-study-{ws_name}.json','w') as f:
            json.dump(all_prompt_ranking, f, ensure_ascii=False, indent=4)

        # assign score for each model as per their rank
        points = {k:0 for k in all_prompt_ranking[0]}
        for each_ranking in all_prompt_ranking:
            for pos,model_name in enumerate(each_ranking):
                points[model_name] += points_for_rank[pos]
        # Calculate win rates
        points = {model: total_points / len(all_prompt_ranking) for model, total_points in points.items()}
        print(f'The aggregated score for user-study dataset at workshop {ws_name} is {points}')

import numpy as np        
pref_score = [0.63, 0.48, 0.52]
# Defining the two new lists
abs_score = [0.73, 0.87, 0.78]
crs_score = [0.63, 0.65, 0.7]

for score in [pref_score,abs_score,crs_score]:
    # Calculating mean and standard deviation for the first list
    mean = np.mean(score)
    std_dev = np.std(score) # Using sample standard deviation (N-1)
    print(f'mean: {mean} and std :{std_dev}')

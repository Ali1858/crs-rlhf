A user study was carried out with 12 participants, organized into three groups. Since the process of generating text with Large Language Models (LLMs) is stochastic, meaning the output can vary even with identical prompts, we aimed to reduce this variability. We generated three responses from the same prompt and gathered feedback from each of the three corresponding user study groups.

To eliminate any potential bias from the order of the answers, we randomized the responses for each question. The mapping keys were stored in a file named `all_keys.json`.

1. **File Naming for Responses**: Files named `humaneval_p_09_t_08_run_X.json` detail the specific parameters (`kwargs`) used for generating responses. Here, `X` represents the run number associated with each user study group, indicating which group's feedback it corresponds to.

2. **Individual Feedback Storage**: The `original-user-study-ws-run-N.json` files capture feedback from each participant. This setup ensures that individual perspectives are documented before any aggregation or analysis.

3. **Aggregated Feedback**: The `final_user-study-ws-run-N.json` files contain the consolidated feedback for each prompt. We employed Tideman's ranking method to merge feedback from all participants, providing a fair and balanced aggregation of their preferences.

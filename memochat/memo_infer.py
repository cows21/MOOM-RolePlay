import os
import json
import torch
import re
from random import sample
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSeq2SeqLM

q_pre = "<s>\n"
qa_link = "\n"
MaxLen = 2048
TarLen = 512
TaskTarLen = {
    "chatting_dialogsum": MaxLen,
    "chatting_alpacagpt4": MaxLen,
    "writing_topiocqa": TarLen // 2,
    "writing_dialogsum": TarLen,
    "retrieval_dialogsum": 32,
    "retrieval_topiocqa": 32
}

def normalize_chatting_outputs(model_outputs):
    """post processing function of chatting response"""
    def white_space_fix(text):
        lines = text.split("\n")
        result = []
        for line in lines:
            result.append(' '.join(line.split()))
        output = '\n'.join(result)
        return output
    return white_space_fix(model_outputs)


@torch.inference_mode()
def get_model_answers(model, tokenizer, local_check, ques_jsons, prompts):
    output_data = []
    for d in ques_jsons:
        new_d = d

        history = {
            "Recent Dialogs": ["user: Hi!", "bot: Hi! How can I help you today?"], 
            "Related Topics": [], 
            "Related Summaries": [], 
            "Related Dialogs": [], 
            "User Input": "",
        }
        memo = {
            "NOTO": [{"summary": "None of the others.", "dialogs": []}]
        }

        for l_i in range(len(new_d["conversations"])):
            if l_i % 2 == 1:
                bot_thinking = {"retrieval": "", "summarization": ""}
                print("=" * 20 + "start of turn {}".format(l_i // 2 + 1) + "=" * 20)
                user = "user: " + new_d["conversations"][l_i - 1]["value"]
                print(user + "\n\n")

                # create summary if recent dialogs exceed threshold
                if len(" ### ".join(history["Recent Dialogs"]).split(" ")) > (MaxLen // 2) or len(history["Recent Dialogs"]) >= 10:
                    history, memo, bot_thinking = run_summary(history, model, tokenizer, memo, local_check, bot_thinking, prompts)

                # retrieve most related topics for every new user input
                history["User Input"] = user
                if len(memo.keys()) > 1:
                    history, bot_thinking = run_retrieval(history, model, tokenizer, memo, local_check, bot_thinking, prompts)
                
                # generate bot response
                system_insturction = prompts["chatting"]["system"]
                task_instruction = prompts["chatting"]["instruction"]
                task_case = "```\nRelated Evidences:\n" + "\n".join(["({}) {}".format(r_tsd_i + 1, {
                                "Related Topics": history["Related Topics"][r_tsd_i], 
                                "Related Summaries": history["Related Summaries"][r_tsd_i], 
                                "Related Dialogs": history["Related Dialogs"][r_tsd_i]
                            }) for r_tsd_i in range(len(history["Related Topics"]))]) + "\n\nRecent Dialogs:\n" + \
                            " ### ".join([hrd.replace("\n", " ") for hrd in history["Recent Dialogs"]]) + "\n```\n\nUser Input:\n" + history["User Input"] + " ### bot: "
                qs = q_pre + system_insturction + task_case + task_instruction + qa_link
                print('this is qs')
                print(qs)
                outputs = gen_model_output(model, tokenizer, qs, local_check, "chatting_dialogsum")
                outputs = normalize_chatting_outputs(outputs)
                history["Recent Dialogs"] += [user, "bot: " + outputs]
                print("bot: " + outputs + "\n")
                print("=" * 20 + "end of turn {}".format(l_i // 2 + 1) + "=" * 20)
                print("\n\n\n\n")
                new_d["conversations"][l_i]["thinking"] = json.dumps(bot_thinking)
                new_d["conversations"][l_i]["value"] = outputs

        output_data.append(d)
    return output_data

def run_eval(model, tokenizer, local_check, question_file, answer_file, prompt_path):
    # get_model_answers(model, tokenizer, num_gpus, local_check, load_in_8bit, ques_jsons, prompts)
    prompts = json.load(open(prompt_path, "r"))

    # split question file into num_gpus files
    ques_jsons = json.load(open(os.path.expanduser(question_file), "r"))

    chunk_size = 1
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ques_i = ques_jsons[i: i + chunk_size]
        ans_handles.append(
            get_model_answers(
                model, tokenizer, local_check, ques_i, prompts
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ans_handle)

    json.dump(ans_jsons, open(os.path.expanduser(answer_file), "w"), indent=2)

    return ans_jsons

def run_summary(history, model, tokenizer, memo, local_check, bot_thinking, prompts):
    """We assume there's no too long input from user, e.g. over 1500 tokens"""
    system_insturction = prompts["writing_dialogsum"]["system"]
    task_instruction = prompts["writing_dialogsum"]["instruction"]
    history_log = "\n\n```\nTask Conversation:\n" + "\n".join(["(line {}) {}".format(h_i + 1, h.replace("\n", " ")) for h_i, h in enumerate(history["Recent Dialogs"][2:])])
    qs = q_pre + system_insturction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + history_log + "\n```" + task_instruction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + qa_link
    print("#" * 20 + "summarizing" + "#" * 20)
    print(qs)
    print("#" * 20 + "summarizing" + "#" * 20)
    sum_history = gen_model_output(model, tokenizer, qs, local_check, "writing_dialogsum")
    sum_history = normalize_model_outputs(sum_history)
    print("#" * 20 + "summarization" + "#" * 20)
    print(sum_history)
    print("#" * 20 + "summarization" + "#" * 20)
    for s in sum_history:
        memo[s["topic"]] = memo.get(s["topic"], []) + [{"summary": s["summary"], "dialogs": history["Recent Dialogs"][2:][(s["start"] - 1):s["end"]]}]
    if local_check:
        memo["test_topic{}".format(len(memo.keys()))] = [{"summary": "test_summary{}".format(len(memo.keys())), "dialogs": history["Recent Dialogs"][2:][2:4]}]
    if len(sum_history) == 0:
        si_0, si_1 = sample(list(range(len(history["Recent Dialogs"][2:]))), 2)
        memo["NOTO"].append({"summary": "Partial dialogs about: {} or {}.".format(history["Recent Dialogs"][2:][si_0], history["Recent Dialogs"][2:][si_1]), "dialogs": history["Recent Dialogs"][2:]})
    history["Recent Dialogs"] = history["Recent Dialogs"][-2:]
    bot_thinking["summarization"] = {"input": qs, "output": sum_history}
    return history, memo, bot_thinking

def gen_model_output(model, tokenizer, input_qs, local_check, task_type):
    if local_check:
        from faker import Faker
        fake = Faker(locale="en")
        return fake.text(2000)
    if "writing" in task_type:
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.encode("]", add_special_tokens=False)[0]]
    elif "retrieval" in task_type:
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[0], tokenizer.encode(" ", add_special_tokens=False)[0]]
    else:
        eos_token_ids = [tokenizer.eos_token_id]
    input_ids = tokenizer([input_qs], max_length=(MaxLen - TarLen), truncation=True, add_special_tokens=False).input_ids
    target_len = min(len(input_ids[0]) + TaskTarLen[task_type], MaxLen)
    repetition_penalty_value = 1.0
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.2,
        max_length=target_len,
        eos_token_id=eos_token_ids,
        repetition_penalty=repetition_penalty_value
    )

    output_ids = output_ids[0][len(input_ids[0]):]
    model_outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return model_outputs

def normalize_model_outputs(model_text):
    """post processing function of memo writing task"""
    extracted_elements = [re.sub(r'\s+', ' ', mt.replace('"', '').replace("'", "")) for mt in re.findall(r"'[^']*'|\"[^\"]*\"|\d+", model_text)]
    model_outputs = []
    ti = 0
    while ti + 7 < len(extracted_elements):
        if extracted_elements[ti] == "topic" and extracted_elements[ti + 2] == "summary" and extracted_elements[ti + 4] == "start" and extracted_elements[ti + 6] == "end":
            try:
                model_outputs.append({"topic": extracted_elements[ti + 1], "summary": extracted_elements[ti + 3], "start": int(extracted_elements[ti + 5]), "end": int(extracted_elements[ti + 7])})
            except:
                pass
        ti += 1
    return model_outputs

def run_retrieval(history, model, tokenizer, memo, local_check, bot_thinking, prompts):
    topics = []
    for k, v in memo.items():
        for vv in v:
            topics.append((k, vv["summary"], vv["dialogs"]))
    system_insturction = prompts["retrieval"]["system"]
    task_instruction = prompts["retrieval"]["instruction"]
    task_case = "```\nQuery Sentence:\n" + history["User Input"][6:] + "\nTopic Options:\n" + \
                "\n".join(["({}) {}".format(v_i + 1, v[0] + ". " + v[1]) for v_i, v in enumerate(topics)]) + "\n```"
    qs = q_pre + system_insturction.replace("OPTION", str(len(topics))) + task_case + task_instruction.replace("OPTION", str(len(topics))) + qa_link
    print("#" * 20 + "retrieving" + "#" * 20)
    print(qs)
    print("#" * 20 + "retrieving" + "#" * 20)
    outputs = gen_model_output(model, tokenizer, qs, local_check, "retrieval_dialogsum")
    print("#" * 20 + "retrieval" + "#" * 20)
    print(outputs)
    print("#" * 20 + "retrieval" + "#" * 20)
    outputs = outputs.split("#")
    chosen_topics = []
    for output in outputs:
        try:
            index_ = int(output) - 1
        except:
            continue
        if index_ < len(topics) and "NOTO" not in topics[index_][0]:
            chosen_topics.append(topics[index_])
    if local_check:
        chosen_topics = sample(topics, min(len(topics) - 1, 2))
    if len(chosen_topics) > 0:
        history["Related Topics"] = [ct[0] for ct in chosen_topics]
        history["Related Summaries"] = [ct[1] for ct in chosen_topics]
        history["Related Dialogs"] = [" ### ".join(ct[2]) for ct in chosen_topics]
    else:
        history["Related Topics"] = []
        history["Related Summaries"] = []
        history["Related Dialogs"] = []
    bot_thinking["retrieval"] = {"input": qs, "output": outputs}
    return history, bot_thinking

if __name__ == "__main__":
    local_check = False
    if not local_check:
        model_id_combine = '/nvme/lisongling/models/Qwen1.5-14B-Chat'
        model_combine = AutoModelForCausalLM.from_pretrained(model_id_combine, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        tokenizer_combine = AutoTokenizer.from_pretrained(model_id_combine, trust_remote_code=True)
    else:
        model_combine = None
        tokenizer_combine = None
    # run_eval(model, tokenizer, local_check, question_file, answer_file, prompt_path)
    question_file_path = 'model/memochat/question_chinese_test.json'
    answer_file_path = 'model/memochat/answer_file_test.json'
    prompt_file_path = 'model/memochat/memo_chat_prompt_chinese.json'
    run_eval(model=model_combine, tokenizer=tokenizer_combine, local_check=local_check, question_file=question_file_path, 
             answer_file=answer_file_path, prompt_path=prompt_file_path)
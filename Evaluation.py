class LLMsEvaluation():
    def get_llm_answers(self, model, tokenizer):
        inputs = tokenizer.apply_chat_template(
            tokenize=True,
            add_generation_prompt=True,  # In case of a CPT Test the model will just predict the next token
            return_tensors="pt",
        ).to("cuda")
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id
        )
        prompt_length = inputs.shape[1]
        response_ids = outputs[0][prompt_length:]
        llm_answer = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return llm_answer
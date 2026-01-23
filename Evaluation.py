class LLMsEvaluation():
    def get_llm_answers(self, model, tokenizer):
        """
        This function takes as input the model and tokenizer directly from the Google colab notebooks,
        run the inference
        Returns the model's answer
        """
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
    def generate_insights(self, dataframe):
        """
        This function takes the generated dataframe with scores
        Generates a graph which shows the model scores and means for every test
        """
        pass # TODO génère d'abord la fonction pour le dataframe je vais écrire celle ci en fonction d'elle
class LLMsEvaluation():
    def one_task_execution(task, promptset):
        """this function is use to execute a specific task at the moment"""
        
        if task not in promptset:
            raise ValueError(f"Task {task} not found in the promptset.")
        
        print (f"Executing task: {task}")
        
        items_list = promptset[task]
        
        #defining instructions for tasks 
        default_instructions = {
            "T1": "Complète le texte suivant de manière cohérente et naturelle en utilisant des mots en camfranglais",
            "T4": "Explique ces concepts de manière simple en utilisant du français standard",
            "Default": "Traite les questions suivantes"
        }
        
        instruction = default_instructions.get(task, default_instructions["Default"])
        
        for i, item in enumerate(items_list):
            if isinstance(item, str):
                """
                here we affect the instruction to his specific task,
                for example for T1 we will use the instruction defined in the default_instructions dict
                """
                instruction_prompt =  instruction
                input_prompt = item 
            
            else :
                instruction_prompt = item.get("instruction", instruction)
                input_prompt = item.get("input", "")

            final_prompt = f"Instruction: {instruction_prompt}\nInput: {input_prompt}"
            
            
                
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
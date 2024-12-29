import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_embed_with_cache_kv=False,
        return_intermediates=False,
    ):
        # Standard GPT-2 forward pass
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache
        )

        logits = outputs.logits
        next_past_key_values = outputs.past_key_values

        # Optional: Compute embeddings from the model's input
        if inputs_embeds is not None:
            embeds = inputs_embeds
        else:
            embeds = self.transformer.wte(input_ids)

        if return_embed_with_cache_kv:
            return embeds, next_past_key_values

        if return_intermediates:
            return logits, embeds, next_past_key_values

        return logits

    @classmethod
    def from_pretrained_with_tokenizer(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load the model and tokenizer together.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
        
        # Set the pad_token to eos_token to avoid padding issues
        tokenizer.pad_token = tokenizer.eos_token
        
        model = cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return model, tokenizer

    def preprocess_input(self, inputs, tokenizer):
        """
        Handle input preprocessing: string list or tensor.
        """
        if isinstance(inputs, list):
            return tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def generate_logits_with_cached_kv(self, input_ids, cached_kv, mask=None):
        """
        Generate logits for the last token using cached key-values.
        """
        outputs = self(
            input_ids=input_ids,
            past_key_values=cached_kv,
            attention_mask=mask
        )
        return outputs

    def generate_embed_with_cached_kv(self, input_ids, cached_kv, mask=None):
        """
        Generate embeddings and updated cached key-values.
        """
        embeds, next_cached_kv = self(
            input_ids=input_ids,
            past_key_values=cached_kv,
            attention_mask=mask,
            return_embed_with_cache_kv=True
        )
        return embeds, next_cached_kv

    def generate_logits_embeds_cached_kv(self, input_ids, mask=None):
        """
        Generate logits, embeddings, and cached key-values for intermediate outputs.
        """
        logits, embeds, cached_kv = self(
            input_ids=input_ids,
            attention_mask=mask,
            return_intermediates=True
        )
        return logits, embeds, cached_kv

# Example usage
if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = CustomGPT2LMHeadModel.from_pretrained_with_tokenizer("gpt2")

    # Input examples
    prompt = "The quick brown fox"
    begin_thought = "jumps over the lazy dog"
    inputs = [prompt, begin_thought]
    processed_inputs = model.preprocess_input(inputs, tokenizer)

    # Forward pass with cached key-values
    output = model(
        input_ids=processed_inputs["input_ids"],
        attention_mask=processed_inputs["attention_mask"],
        return_intermediates=True
    )

    logits, embeds, cached_kv = output
    print("Logits shape:", logits.shape)
    print("Embeddings shape:", embeds.shape)
    print("Cached KV length:", len(cached_kv))

    # Example: Generate logits for the last token using cached key-values
    last_token_logits = model.generate_logits_with_cached_kv(
        input_ids=processed_inputs["input_ids"][:, -1:],  # Last token
        cached_kv=cached_kv
    )
    print("Last token logits shape:", last_token_logits.shape)

    # Example: Generate embeddings with cached key-values
    latent_token = processed_inputs["input_ids"][:, :1]  # First token
    latent_embeds, updated_cached_kv = model.generate_embed_with_cached_kv(
        input_ids=latent_token,
        cached_kv=cached_kv
    )
    print("Latent embeddings shape:", latent_embeds.shape)
    print("Updated cached KV length:", len(updated_cached_kv))

    # Example: Generate logits, embeddings, and cached key-values for intermediate outputs
    prompt_logits, prompt_embeds, prompt_cached_kv = model.generate_logits_embeds_cached_kv(
        input_ids=processed_inputs["input_ids"],
        mask=processed_inputs["attention_mask"]
    )
    print("Prompt logits shape:", prompt_logits.shape)
    print("Prompt embeddings shape:", prompt_embeds.shape)
    print("Prompt cached KV length:", len(prompt_cached_kv))
    
    
# answer_logits = self.model(out[:, -1:], cached_kv = cached_kv)
# latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, mask = mask, return_embed_with_cache_kv = True)
# latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, mask = mask, return_embed_with_cache_kv = True)
# prompt_logits, embeds, cached_kv = self.model([prompt, begin_thought], mask=mask, return_intermediates=True)
# logits = self.model(final_forward, cached_kv=cached_kv, mask=mask)
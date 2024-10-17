# Setup

## Changes in config.py
- Add the name of the model you're using as the base model. If you are trying to serve a local model, you must first run an OpenAI compatible server with the model running - like [vLLM](https://docs.vllm.ai/en/v0.6.0/serving/openai_compatible_server.html)
- Update (if needed) the port and base_url location
- If it's the name of a propreitory model, then just mention the model name and make sure your API keys are in your .env file

## Changes to make in modules.py
- If you want to run proprietary models such as GPT-4o-x, claude-3-x etc, then make sure to not pass the base_url, api_key to the 'Model' class
- Run with the query you want
- Adjust parameters like max_depth, max_children, beam_width etc as per your wish

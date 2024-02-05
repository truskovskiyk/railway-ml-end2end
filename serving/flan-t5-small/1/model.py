import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from pprint import pprint


class TritonPythonModel:
    def initialize(self, config):
        pprint(config)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        # self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    def execute(self, requests):
        responses = []
        for request in requests:
            schema = pb_utils.get_input_tensor_by_name(request, "schema").as_numpy().tolist()[0].decode("utf-8")
            user_query = pb_utils.get_input_tensor_by_name(request, "user_query").as_numpy().tolist()[0].decode("utf-8")

            # inputs = self.tokenizer(input_text, return_tensors="pt")

            # with torch.inference_mode():
            #     outputs = self.model.generate(**inputs)

            # generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            generated_text = "test"

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("generated_ql", np.array([generated_text], dtype=np.object_))]
            )
            responses.append(inference_response)
        return responses

import tritonclient.http as httpclient
import numpy as np
import typer


def generate_text(schema: str, user_query: str, url):
    try:
        triton_client = httpclient.InferenceServerClient(url=url)

        schema_data = np.array([schema], dtype=object)
        user_query_data = np.array([user_query], dtype=object)

        inputs = [
            httpclient.InferInput("schema", schema_data.shape, "BYTES"),
            httpclient.InferInput("user_query", user_query_data.shape, "BYTES"),
        ]

        # Set the input data
        inputs[0].set_data_from_numpy(schema_data)
        inputs[1].set_data_from_numpy(user_query_data)

        # Make the inference request
        response = triton_client.infer(model_name="flan-t5-small", inputs=inputs)
        print(response)
        # Extract the output data from the response
        generated_ql = response.as_numpy("generated_ql")

        print(f"generated_ql = {generated_ql}")

    except Exception as e:
        print("Exception: {}".format(e))


def cli():
    app = typer.Typer()
    app.command()(generate_text)
    app()


if __name__ == "__main__":
    cli()
    # Example usage
    # url = 'adorable-silver-production.up.railway.app'
    # input_text = "A step by step recipe to make bolognese pasta:"
    # result = generate_text("A step by step recipe to make bolognese pasta:", url='https://adorable-silver-production.up.railway.app/')
    # print("Generated Text: ", result)

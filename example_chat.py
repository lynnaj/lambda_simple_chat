import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    # Bedrock Runtime client used to invoke and question the models
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        endpoint_url = 'https://vpce-xxxx.bedrock-runtime.us-x-region.vpce.amazonaws.com'
        )   
    prompt = event['prompt']
    model_id = "amazon.nova-pro-v1:0" # Replace with your Bedrock model ID

    # Define your system prompt(s).
    system_list = [
            {
                "text": "You are a help assistant."
            }
    ]

    # Define one or more messages using the "user" and "assistant" roles.
    message_list = [{"role": "user", "content": [{"text": prompt}]}]

    # Configure the inference parameters.
    inf_params = {"maxTokens": 500, "topP": 0.9, "topK": 20, "temperature": 0.7}

    try:
        response = bedrock_runtime.invoke_model(
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": message_list,
                "system": system_list,
                "inferenceConfig": inf_params
            }),
            modelId= model_id,
            contentType="application/json",
            accept="application/json"
        )

        logger.info(f"Raw Bedrock response: {response}")

        response_body = json.loads(response.get('body').read())

        logger.info(f"json response: {response_body}")
        # The response from the model now mapped to the answer
        answer = response_body['output']['message']['content'][0]['text']

        return {
            'statusCode': 200,
            'body': json.dumps({'response': answer})
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

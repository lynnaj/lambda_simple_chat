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
    model_id = "amazon.titan-text-express-v1"

    native_request = {
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.5,
        },
    }
    try:
        response = bedrock_runtime.invoke_model(
            body=json.dumps(native_request),
            modelId= model_id
            )

        logger.info(f"Raw Bedrock response: {response}")

        response_body = json.loads(response.get('body').read())

        logger.info(f"json response: {response_body}")
        # The response from the model now mapped to the answer
        answer = response_body['results'][0]['outputText']

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

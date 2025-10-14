# AWS Bedrock Configuration Guide

This guide explains how to configure your backend to use AWS Bedrock Anthropic Claude Haiku 3 instead of OpenAI.

## Prerequisites

1. AWS Account with Bedrock access
2. EC2 instance with appropriate IAM role (recommended) OR AWS credentials
3. Access to Anthropic Claude models in AWS Bedrock

## Configuration Steps

### 1. Environment Variables

Update your `.env` file with the following configuration:

```env
# Set LLM provider to bedrock
llm_provider=bedrock

# AWS Configuration
aws_region=us-east-1
aws_access_key_id=your_aws_access_key_id  # Optional if using IAM role
aws_secret_access_key=your_aws_secret_access_key  # Optional if using IAM role

# S3 Configuration (required for file uploads)
bucket_name=your_s3_bucket_name

# Database and other configurations remain the same
mongo_db_url=mongodb://localhost:27017/mindmap
```

### 2. IAM Role Configuration (Recommended for EC2)

If deploying on EC2, create an IAM role with the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "textract:StartDocumentAnalysis",
                "textract:GetDocumentAnalysis"
            ],
            "Resource": "*"
        }
    ]
}
```

### 3. AWS Bedrock Model Access

Ensure you have access to the following models in AWS Bedrock:
- `anthropic.claude-3-haiku-20240307-v1:0` (for text generation)
- `amazon.titan-embed-text-v1` (for embeddings)

To request access:
1. Go to AWS Bedrock console
2. Navigate to "Model access" in the left sidebar
3. Request access to Anthropic Claude 3 Haiku and Amazon Titan Embed models

### 4. S3 Bucket Setup

Create an S3 bucket for file storage:
1. Create a new S3 bucket in your preferred region
2. Configure appropriate bucket policies for your use case
3. Update the `bucket_name` in your `.env` file

### 5. Testing the Configuration

To test if AWS Bedrock is working correctly:

1. Start your backend server
2. Check the logs for any AWS authentication errors
3. Try uploading a document and generating a summary
4. The system should now use Claude Haiku 3 instead of OpenAI

## Model Configuration

The current configuration uses:
- **LLM Model**: `anthropic.claude-3-haiku-20240307-v1:0`
- **Embedding Model**: `amazon.titan-embed-text-v1`
- **Region**: Configurable via `aws_region` environment variable

## Switching Between Providers

You can easily switch between OpenAI and AWS Bedrock by changing the `llm_provider` environment variable:

- `llm_provider=openai` - Uses OpenAI GPT-4o
- `llm_provider=bedrock` - Uses AWS Bedrock Claude Haiku 3

## Limitations

When using AWS Bedrock:
1. SQL and CSV bots still use OpenAI (Vanna library limitation)
2. Some advanced OpenAI Assistant API features are replaced with direct LLM calls
3. File processing may have different token limits compared to OpenAI

## Troubleshooting

### Common Issues:

1. **Authentication Error**: Ensure IAM role has correct permissions or AWS credentials are valid
2. **Model Access Denied**: Request access to Claude models in Bedrock console
3. **Region Mismatch**: Ensure all AWS services are in the same region
4. **S3 Permissions**: Verify S3 bucket permissions for read/write access

### Debug Steps:

1. Check AWS credentials: `aws sts get-caller-identity`
2. Verify Bedrock access: Try invoking a model via AWS CLI
3. Test S3 access: Upload a test file to your bucket
4. Check application logs for specific error messages

## Cost Considerations

AWS Bedrock pricing is different from OpenAI:
- Claude Haiku 3: Pay per input/output tokens
- Titan Embeddings: Pay per input tokens
- S3 storage and Textract usage also apply

Monitor your AWS costs through the AWS Cost Explorer.
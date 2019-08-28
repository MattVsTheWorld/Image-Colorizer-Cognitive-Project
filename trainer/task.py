from trainer.train import run


def main():
    run()


if __name__ == '__main__':
    main()


'''
gcloud ai-platform jobs submit training test_job --job-dir gs://images_data/cs-job-dir --runtime-version 1.13 --module-name trainer.task --package-path C:\Users\tomlo\Desktop\Cognitive-Project\trainer --region capra --config=C:\Users\tomlo\Desktop\Cognitive-Project\trainer\cloudml-gpu.yaml --stream-logs
'''

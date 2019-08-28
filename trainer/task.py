from trainer.train import run


def main():
    run()


if __name__ == '__main__':
    main()


'''
Cambiare il nome del job aumentando di uno il numero e il path alla cartella trainer e al file yaml
gcloud ai-platform jobs submit training test_jo12 --job-dir gs://images_data/cs-job-dir/ --runtime-version 1.13 --module-name trainer.task --package-path C:\Users\tomlo\Desktop\Cognitive-Project\trainer --region us-central1 --config=C:\Users\tomlo\Desktop\Cognitive-Project\trainer\cloudml-gpu.yaml --stream-logs
'''

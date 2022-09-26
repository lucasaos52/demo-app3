gcloud builds submit --tag gcr.io/portfoliosimulator-363718/portfoliosimulator  --project=portfoliosimulator-363718

gcloud run deploy --image gcr.io/portfoliosimulator-363718/portfoliosimulator --platform managed  --project=portfoliosimulator-363718 --allow-unauthenticated
import neptune
import os

from dotenv import load_dotenv
from time import sleep

# Load .env as environment variables
load_dotenv()
api_token = os.environ.get('NEPTUNE_TOKEN')
proj_name = os.environ.get('PROJECT_NAME')

neptune.init(project_qualified_name=proj_name,
             api_token=api_token)

# Create experiment
neptune.create_experiment(name='TESTING 123')

# Log metrics to experiment

neptune.log_metric('single_metric', 0.62)

for i in range(100):
    sleep(0.2)  # to see logging live
    neptune.log_metric('random_training_metric', i * 0.6)
    neptune.log_metric('other_random_training_metric', i * 0.4)

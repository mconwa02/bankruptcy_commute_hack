from sklearn.model_selection import train_test_split
from utils import get_configured_logger

from data_preprocess import read_data
from modelling_config import random_forest_tuning, stratified_sampling
from modelling_config import random_forest_model, model_metrics

logger = get_configured_logger(__name__)

df = read_data("train_dataset.csv")

data = df.drop(['bankruptcy_label'], axis=1)
target = df['bankruptcy_label']

x_sample, _, y_sample, _ = stratified_sampling(data, target)

x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    test_size=0.25,
                                                    random_state=1,
                                                    stratify=target)

logger.info(f"x train shape {x_train.shape}")
logger.info(f"x test shape {x_test.shape}")
logger.info(f"y train shape {y_train.shape}")
logger.info(f"y test shape {x_test.shape}")

rf_model = random_forest_model()
rf_model.fit(x_train, y_train)

logger.info(rf_model)

model_metrics(x_train, y_train, rf_model)

# to set hyper-parameters for the  random forest
# cv_rfc = random_forest_tuning(data, target)
# best_rf = cv_rfc.best_parmas_
# logger.info(f"Best Random Forest Parameter {best_rf}")


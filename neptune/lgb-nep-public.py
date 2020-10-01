import lightgbm as lgb
import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

# Connect your script to Neptune
neptune.init(api_token='ANONYMOUS',
             project_qualified_name='shared/showroom')

PARAMS = {'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'bagging_fraction': 0.7,
          'seed': 2020,
          }
# NUM_BOOSTING_ROUNDS = 10


# prefix='fold{}_'.format(fold_id)

# Create an experiment and log hyperparameters
exp = neptune.create_experiment(name='test-lgb-1',
                                description='1st test on LC data, train-test-split, basic-fe',
                                params={**PARAMS,
                                        # 'num_boosting_round': NUM_BOOSTING_ROUNDS
                                        },
                                upload_source_files=['train.py', 'environment.yaml'],
                                )

# read data
train = pd.read_csv('train.csv')
test = pd.read_csv('testa.csv')

fea = [f for f in train.columns if f not in ['id', 'isDefault']]
X_train = train[fea]
X_test = test[fea]
y_train = train['isDefault']

folds = 5
seed = 46
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
neptune.log_metric('kfold-seed', seed)

# _______________________________


trn_cv = []
val_cv = []

for i, (train_ind, valid_ind) in enumerate(kf.split(X_train, y_train)):
    print('************************************ {} ************************************'.format(str(i + 1)))
    X_trn, y_trn, X_val, y_val = X_train.iloc[train_ind], y_train[train_ind], X_train.iloc[valid_ind], y_train[valid_ind]

    train_matrix = lgb.Dataset(X_trn, label=y_trn)  # categorical_feature = ['grade, subGrade']
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    prefix = 'fold{}_'.format(i)

    gbm = lgb.train(PARAMS,
                    train_set=train_matrix,
                    # num_boost_round=NUM_BOOSTING_ROUNDS,
                    valid_sets=[train_matrix, valid_matrix],
                    valid_names=['train', 'eval'],
                    verbose_eval=200,
                    early_stopping_rounds=200,
                    callbacks=[neptune_monitor(exp, prefix)],  # monitor learning curves (prefix)
                    )

    val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    val_cv.append(roc_auc_score(y_val, val_pred))
    print(val_cv)

    # # save and log model
    # gbm.save_model('model.txt')
    # neptune.send_artifact('model.txt')

val_mean = np.mean(val_cv)
val_std = np.std(val_cv)
neptune.log_metric('val_mean', val_mean)
neptune.log_metric('val_std', val_std)


# 模型加载
# gbm = lgb.Booster(model_file='model.txt')
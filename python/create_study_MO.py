import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import HyperbandPruner

journal_storage = 1
url = 'sqlite:///BO_Search_learn_1.sqlite'
url_journal = "./journal_learn_1.log"
studyname = "BO_Search_learn"

if journal_storage == 1:
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(url_journal),
    )
else:
    storage = optuna.storages.RDBStorage(
        url=url,
        heartbeat_interval=30,
        grace_period=60,
        #failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        engine_kwargs={"pool_size": 96*4,"max_overflow" : 0, "connect_args": {"timeout": 360}},
        #skip_table_creation={True}        
        )
    
# Hinzufügen des Hyperband Pruners
pruner = HyperbandPruner()
     
study = optuna.create_study(sampler=optuna.samplers.TPESampler(constant_liar=True), directions=["minimize", "minimize"],
        storage=storage, study_name=studyname, pruner=pruner)
    
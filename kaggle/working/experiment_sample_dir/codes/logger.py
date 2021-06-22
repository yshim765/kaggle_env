import mlflow


class Logger():
    def __init__(self):
        self.run = mlflow.active_run()

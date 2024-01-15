import os
import time
import numpy as np
import wandb
import torch
from schnetpack.train.hooks import Hook


__all__ = ["LoggingHook", "WandBHook", "CSVHook", "TensorboardHook", "PrintingHook"]

'''
class PlotDistMetricsHook(Hook):
     """Base class for saving plots of distributions of various metric errors.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
    ):
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_path = log_path

        self._train_loss = 0
        self._counter = 0
        self.metrics = metrics
'''

class LoggingHook(Hook):
    """Base class for logging hooks.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        log_gradients (bool, optional): enable logging of the mean values of first and last layer of gradients

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        log_gradients = True,
    ):
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_path = log_path

        self._train_loss = 0
        self._weight_first_layer = 0
        self._weight_last_layer = 0
        self._grad_first_layer = 0
        self._grad_last_layer = 0
        
        self._counter = 0
        self._grad_counter = 0
        self.metrics = metrics

    def on_epoch_begin(self, trainer):
        """Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.

        """
        # reset train_loss and counter
        if self.log_train_loss:
            self._train_loss = 0.0
            self._counter = 0
            
        else:
            self._train_loss = None
        
        if self.log_gradients:
            self._weight_first_layer = 0.0
            self._weight_last_layer = 0.0
            self._grad_first_layer = 0.0
            self._grad_last_layer = 0.0
            self._grad_counter = 0
            
        else:
            self._weight_first_layer = None
            self._weight_last_layer = None
            self._grad_first_layer = None
            self._grad_last_layer = None
        

    def on_batch_end(self, trainer, train_batch, result, loss):
        if self.log_train_loss:
            n_samples = self._batch_size(result)
            self._train_loss += float(loss.data) * n_samples
            self._counter += n_samples

    def _batch_size(self, result):
        if type(result) is dict:
            n_samples = list(result.values())[0].size(0)
        elif type(result) in [list, tuple]:
            n_samples = result[0].size(0)
        else:
            n_samples = result.size(0)
        return n_samples

    def on_validation_begin(self, trainer):
        for metric in self.metrics:
            metric.reset()

    def on_validation_batch_end(self, trainer, val_batch, val_result):
        for metric in self.metrics:
            metric.add_batch(val_batch, val_result)


class PrintingHook(Hook):
    """Base class for priting statements.

    Args:
    
    """
    
    def __init__(self, spherical_harmonics = False, vanilla_NN = False):
        self.spherical_harmonics = spherical_harmonics
        self.vanilla_NN = vanilla_NN
    
    def on_train_begin(self, trainer):
        print("Printing Hook present")
    
    
    def on_batch_end(self, trainer, train_batch, result, loss):
        
        
        if self.spherical_harmonics:
            weight_first_layer = trainer._model.representation.interactions[0].filter_network_sblock[0].weight
        elif self.vanilla_NN is False:
            weight_first_layer = trainer._model.representation.interactions[0].filter_network[0].weight
            
        if self.vanilla_NN:
            weight_first_layer = trainer._model.representation.mlp.out_net[0].weight
            
        weight_last_layer = trainer._model.output_modules[0].out_net[-1].out_net[-1].weight
        grad_first_layer = torch.mean(abs(weight_first_layer.grad))
        grad_last_layer = torch.mean(abs(weight_last_layer.grad))
        
        
        print('Trainig loss : ', loss)
        print('Gradient first layer : ', grad_first_layer)
        print('Gradient last layer : ', grad_last_layer)

        batch_grad_norm = 0.0
        batch_norm = 0.0
        for name, p in trainer._model.named_parameters():
            if p.grad is None:
                param_grad_norm = 0.0
            else:
                param_grad_norm = p.grad.data.norm(2)
    
            batch_grad_norm += param_grad_norm ** 2
            batch_norm += p.data.norm(2)**2
        
        batch_grad_norm = batch_grad_norm ** (1. / 2)
        batch_norm = batch_norm ** (1. / 2)
        
        print('Batch gradient norm : ', batch_grad_norm)
        print('Batch norm : ', batch_norm)
        
        print('\n\n')

class WandBHook(LoggingHook):
    """Hook for logging training and validation process to WandB server.

    Args:
        config (dict): wandb config dictionary which contains all the parameters pertaining to the model
        project (str): project name according to the WandB server
        entity (str): entity name according to the WandB server
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        log_gradients (bool, optional): enable logging of the mean values of first and last layer of gradients
        every_n_epochs (int, optional): epochs after which logging takes place.

    """
    
    def __init__(
        self,
        config,
        project,
        entity,
        name,
        spherical_harmonics = False,
        vanilla_NN = False,
        log_path = "",
        metrics = [],
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        log_gradients = True,
        every_batch = False,
        every_n_epochs=1, 
    ):  
        self.spherical_harmonics = spherical_harmonics
        self.vanilla_NN = vanilla_NN
        
        self.config = config
        self.project = project
        self.entity = entity
        self.name = name
        self.log_path = log_path
        self.metrics = metrics
        
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_gradients = log_gradients
        self.every_batch = every_batch
        self.every_n_epochs = every_n_epochs
                
        self.save_dict = {}
        self.save_batch_dict = {}
        
    def on_train_begin(self, trainer):
        for metric in self.metrics:
            metric.reset()
    
    
    def on_batch_end(self, trainer, train_batch, result, loss):
        
        if self.log_train_loss:
            n_samples = self._batch_size(result)
            self._train_loss += float(loss.data) * n_samples
            self._counter += n_samples
                        
        if self.log_gradients:
            if self.spherical_harmonics:
                weight_first_layer = trainer._model.representation.interactions[0].filter_network_sblock[0].weight
            elif self.vanilla_NN is False:
                weight_first_layer = trainer._model.representation.interactions[0].filter_network[0].weight
                
            if self.vanilla_NN is True:
                weight_first_layer = trainer._model.representation.mlp.out_net[0].weight
                
            weight_last_layer = trainer._model.output_modules[0].out_net[-1].out_net[-1].weight
            self._weight_first_layer += torch.mean(abs(weight_first_layer))
            self._weight_last_layer += torch.mean(abs(weight_last_layer))
            self._grad_first_layer += torch.mean(abs(weight_first_layer.grad))
            self._grad_last_layer += torch.mean(abs(weight_last_layer.grad))
            self._grad_counter += 1
           
        if self.every_batch:
            self.save_batch_dict["batch_loss"]  = loss.data
            self.save_batch_dict["batch_grad_first_layer"] = torch.mean(abs(weight_first_layer.grad))
            self.save_batch_dict["batch_grad_last_layer"] = torch.mean(abs(weight_last_layer.grad))
            
            batch_grad_norm = 0.0
            
            for p in trainer._model.parameters():
                if p.grad is None:
                    param_norm = 0.0
                else:
                    param_norm = p.grad.data.norm(2)
        
                batch_grad_norm += param_norm ** 2
            batch_grad_norm = batch_grad_norm ** (1. / 2)
                
            self.save_batch_dict["batch_grad_norm"] = batch_grad_norm
            
            wandb.log(self.save_batch_dict)
            
            if torch.isnan(batch_grad_norm):
                torch.save(self.trainer._model, self.trainer.best_model + "_nan_loss")
            
            
        for metric in self.metrics:
            metric.add_batch(train_batch, result)
    
    def on_train_batches_end(self, trainer):
    
        if trainer.epoch % self.every_n_epochs == 0:
        
            for i, metric in enumerate(self.metrics):
                self.save_dict["training_" + metric.name] = metric.aggregate()
            
         
    def on_validation_end(self, trainer, val_loss):
    
        if trainer.epoch % self.every_n_epochs == 0:

            if self.log_learning_rate:
                self.save_dict["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]

            if self.log_train_loss:
                self.save_dict["training_loss"] =  self._train_loss / self._counter

            if self.log_validation_loss:
                self.save_dict["validation_loss"] = val_loss
                
            if self.log_gradients:
                self.save_dict["weight_first_layer"] = self._weight_first_layer/self._grad_counter
                self.save_dict["weight_last_layer"] = self._weight_last_layer/self._grad_counter
                self.save_dict["grad_first_layer"] = self._grad_first_layer/self._grad_counter
                self.save_dict["grad_last_layer"] = self._grad_last_layer/self._grad_counter

            for i, metric in enumerate(self.metrics):
                self.save_dict["validation_" + metric.name] = metric.aggregate()
        
            
            wandb.log(self.save_dict)
        
        

class CSVHook(LoggingHook):
    """Hook for logging training process to CSV files.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        every_n_epochs=1,
    ):
        log_path = os.path.join(log_path, "log.csv")
        super(CSVHook, self).__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate
        )
        self._offset = 0
        self._restart = False
        self.every_n_epochs = every_n_epochs

    def on_train_begin(self, trainer):

        if os.path.exists(self.log_path):
            remove_file = False
            with open(self.log_path, "r") as f:
                # Ensure there is one entry apart from header
                lines = f.readlines()
                if len(lines) > 1:
                    self._offset = float(lines[-1].split(",")[0]) - time.time()
                    self._restart = True
                else:
                    remove_file = True

            # Empty up to header, remove to avoid adding header twice
            if remove_file:
                os.remove(self.log_path)
        else:
            self._offset = -time.time()
            # Create the log dir if it does not exists, since write cannot
            # create a full path
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if not self._restart:
            log = ""
            log += "Time"

            if self.log_learning_rate:
                log += ",Learning rate"

            if self.log_train_loss:
                log += ",Train loss"

            if self.log_validation_loss:
                log += ",Validation loss"

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                log += str(metric.name)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a+") as f:
                f.write(log + os.linesep)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)

            if self.log_learning_rate:
                log += "," + str(trainer.optimizer.param_groups[0]["lr"])

            if self.log_train_loss:
                log += "," + str(self._train_loss / self._counter)

            if self.log_validation_loss:
                log += "," + str(val_loss)

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, "__iter__"):
                    log += ",".join([str(j) for j in m])
                else:
                    log += str(m)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a") as f:
                f.write(log + os.linesep)


class TensorboardHook(LoggingHook):
    """Hook for logging training process to tensorboard.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.
        img_every_n_epochs (int, optional):
        log_histogram (bool, optional):

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        every_n_epochs=1,
        img_every_n_epochs=10,
        log_histogram=False,
    ):
        from tensorboardX import SummaryWriter

        super(TensorboardHook, self).__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate
        )
        self.writer = SummaryWriter(self.log_path)
        self.every_n_epochs = every_n_epochs
        self.log_histogram = log_histogram
        self.img_every_n_epochs = img_every_n_epochs

    def on_epoch_end(self, trainer):
        if trainer.epoch % self.every_n_epochs == 0:
            if self.log_train_loss:
                self.writer.add_scalar(
                    "train/loss", self._train_loss / self._counter, trainer.epoch
                )
            if self.log_learning_rate:
                self.writer.add_scalar(
                    "train/learning_rate",
                    trainer.optimizer.param_groups[0]["lr"],
                    trainer.epoch,
                )

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            for metric in self.metrics:
                m = metric.aggregate()

                if np.isscalar(m):
                    self.writer.add_scalar(
                        "metrics/%s" % metric.name, float(m), trainer.epoch
                    )
                elif m.ndim == 2:
                    if trainer.epoch % self.img_every_n_epochs == 0:
                        import matplotlib.pyplot as plt

                        # tensorboardX only accepts images as numpy arrays.
                        # we therefore convert plots in numpy array
                        # see https://github.com/lanpa/tensorboard-pytorch/blob/master/examples/matplotlib_demo.py
                        fig = plt.figure()
                        plt.colorbar(plt.pcolor(m))
                        fig.canvas.draw()

                        np_image = np.fromstring(
                            fig.canvas.tostring_rgb(), dtype="uint8"
                        )
                        np_image = np_image.reshape(
                            fig.canvas.get_width_height()[::-1] + (3,)
                        )

                        plt.close(fig)

                        self.writer.add_image(
                            "metrics/%s" % metric.name, np_image, trainer.epoch
                        )

            if self.log_validation_loss:
                self.writer.add_scalar("train/val_loss", float(val_loss), trainer.step)

            if self.log_histogram:
                for name, param in trainer._model.named_parameters():
                    self.writer.add_histogram(
                        name, param.detach().cpu().numpy(), trainer.epoch
                    )

    def on_train_ends(self, trainer):
        self.writer.close()

    def on_train_failed(self, trainer):
        self.writer.close()

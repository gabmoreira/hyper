"""
    trainer.py
    Feb 21 2023
    Gabriel Moreira
"""

import os
import torch
import numpy as np

from tqdm import tqdm

from tracker import Tracker


class Trainer:
    def __init__(
        self,
        model,
        epochs,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        dev_loader,
        device,
        name,
        resume):

        self.model        = model
        self.epochs       = epochs
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.criterion    = criterion
        self.train_loader = train_loader
        self.dev_loader   = dev_loader
        self.device       = device
        self.name         = name
        self.start_epoch  = 1

        self.tracker = Tracker(['epoch',
                                'train_loss',
                                'train_acc', 
                                'dev_loss', 
                                'dev_acc',
                                'lr'], os.path.join('./experiments', name), load=resume)

        if resume:
            self.resume_checkpoint()


    def fit(self):
        """
            Fit model to training set over #epochs
        """
        is_best = False

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss, train_acc = self.train_epoch(epoch)
            dev_loss, dev_acc     = self.validate_epoch()

            self.epoch_verbose(epoch, train_loss, train_acc, dev_loss, dev_acc)
            self.scheduler.step()

            # Check if better than previous models
            if epoch > 1:
                is_best = self.tracker.isLarger('dev_acc', dev_acc)
            else:
                is_best = True

            self.tracker.update(epoch=epoch,
                                train_loss=train_loss,
                                train_acc=train_acc,
                                dev_loss=dev_loss,
                                dev_acc=dev_acc,
                                lr=self.optimizer.param_groups[0]['lr'])

            self.save_checkpoint(epoch, is_best)


    def train_epoch(self, epoch):
        """
            Train model for ONE epoch
        """
        self.model.train()

        batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, desc='Train') 

        total_loss_epoch = 0

        avg_loss_epoch = float('inf')
        total_correct  = 0
        total = 0
        
        for i_batch, batch_dict in enumerate(self.train_loader):
            batch_data   = batch_dict['data'].to(self.device)
            batch_target = batch_dict['target'].to(self.device)
            
            self.optimizer.zero_grad()
        
            batch_features = self.model(batch_data)
            loss_batch     = self.criterion(batch_features, batch_target)  
            
            loss_batch.backward()
            self.optimizer.step()

            total_loss_epoch += loss_batch.detach()
            avg_loss_epoch    = float(total_loss_epoch / (i_batch + 1))
            
            tc, t = self.criterion.scores()
            total_correct += tc
            total += t
                
            batch_bar.set_postfix(
                loss="{:1.5e}".format(avg_loss_epoch),
                acc="{:.4f}".format(total_correct / total),
                lr="{:1.2e}".format(float(self.optimizer.param_groups[0]['lr'])))
            batch_bar.update()

        batch_bar.close()

        return float(avg_loss_epoch), float(total_correct/total)
    
    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_correct  = 0
        total = 0
        
        for i_batch, batch_dict in enumerate(tqdm(self.dev_loader)):
            batch_data   = batch_dict['data'].to(self.device)  
            batch_target = batch_dict['target'].to(self.device)
                
            batch_features = self.model(batch_data)
            loss_batch = self.criterion(batch_features, batch_target)  

            total_loss += loss_batch.detach()
            
            tc, t = self.criterion.scores()
            total_correct += tc
            total += t
                
        return float(total_loss / i_batch), float(total_correct / total)


    def save_checkpoint(self, epoch, is_best):
        '''
            Save model dict and hyperparams
        '''
        checkpoint = {"epoch"     : epoch,
                      "model"     : self.model.state_dict(),
                      "optimizer" : self.optimizer,
                      "scheduler" : self.scheduler }

        # Save checkpoint to resume training later
        checkpoint_path = os.path.join('./experiments', self.name, 'checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        print('Checkpoint saved: {}'.format(checkpoint_path))

        # Save best model weights
        if is_best:
            best_path = os.path.join('./experiments/', self.name, "best_weights.pt")
            torch.save(self.model.state_dict(), best_path)
            print("Saving best model: {}".format(best_path))


    def resume_checkpoint(self):
        '''
        '''
        resume_path = os.path.join('./experiments/', self.name, "checkpoint.pt")
        print("Loading checkpoint: {} ...".format(resume_path))

        checkpoint       = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.optimizer   = checkpoint["optimizer"]
        self.scheduler   = checkpoint["scheduler"]
        self.model.load_state_dict(checkpoint["model"])
        
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    
    def epoch_verbose(self, epoch, train_loss, train_acc, dev_loss, dev_acc):
        log = "\nEpoch: {}/{} summary:".format(epoch, self.epochs)
        log += "\n            Train acc (%) |  {:.4f}".format(train_acc * 100)
        log += "\n            Dev acc   (%) |  {:.4f}".format(dev_acc * 100)
        log += "\n            Train loss    |  {:1.6e}".format(train_loss)
        log += "\n            Dev loss      |  {:1.6e}".format(dev_loss)
        print(log)
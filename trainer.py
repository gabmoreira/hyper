"""
    loss.py
    Mar 4 2023
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
        train_loss,
        val_loss,
        train_loader,
        val_loader,
        val_freq: int,
        device: str,
        name: str,
        resume: bool):

        self.model        = model
        self.epochs       = epochs
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loss   = train_loss
        self.val_loss     = val_loss
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.val_freq     = val_freq
        self.device       = device
        self.name         = name
        self.start_epoch  = 1

        self.tracker = Tracker(['epoch',
                                'train_loss',
                                'train_acc', 
                                'val_loss', 
                                'val_acc',
                                'lr'], os.path.join('./experiments', name), load=resume)

        if resume:
            self.resume_checkpoint()


    def fit(self):
        """
            Fit model to training set over #epochs
        """
        is_best = False

        for epoch in range(self.start_epoch, self.epochs+1):
            train_out = self.train_epoch(epoch)
            
            if epoch % val_freq == 0:
                val_out = self.validate_epoch()
            
            self.epoch_verbose(epoch, **train_out, **val_out)
                
            # Check if better than previous models
            is_best = self.tracker.is_larger('val_acc', val_acc)

            self.tracker.update(epoch=epoch,
                                train_loss=train_loss,
                                train_acc=train_acc,
                                val_loss=val_loss,
                                val_acc=val_acc,
                                lr=self.optimizer.param_groups[0]['lr'])

            self.scheduler.step()
            self.save_checkpoint(epoch, is_best)


    def train_epoch(self, epoch):
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
            loss_batch     = self.train_loss(batch_features, batch_target)  
            
            loss_batch.backward()
            self.optimizer.step()

            total_loss_epoch += loss_batch.detach()
            avg_loss_epoch    = float(total_loss_epoch / (i_batch + 1))
            
            tc, t = self.train_loss.scores()
            total_correct += tc
            total += t
                
            batch_bar.set_postfix(
                loss="{:1.5e}".format(avg_loss_epoch),
                acc="{:.4f}".format(total_correct / total),
                lr="{:1.2e}".format(float(self.optimizer.param_groups[0]['lr'])))
            batch_bar.update()

        batch_bar.close()

        return {'loss' : float(avg_loss_epoch), 'acc' : float(total_correct/total)}
    
    
    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_correct  = 0
        total = 0
        
        for i_batch, batch_dict in enumerate(tqdm(self.val_loader)):
            batch_data   = batch_dict['data'].to(self.device)  
            batch_target = batch_dict['target'].to(self.device)
                
            batch_features = self.model(batch_data)
            loss_batch = self.val_loss(batch_features, batch_target)  

            total_loss += loss_batch.detach()
            
            tc, t = self.val_criterion.scores()
            total_correct += tc
            total += t
                
        return float(total_loss / i_batch), float(total_correct / total)


    def save_checkpoint(self, epoch, is_best):
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
        resume_path = os.path.join('./experiments/', self.name, 'checkpoint.pt')
        print('Loading checkpoint: {} ...'.format(resume_path))

        checkpoint       = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer   = checkpoint['optimizer']
        self.scheduler   = checkpoint['scheduler']
        self.model.load_state_dict(checkpoint["model"])
        
        print('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

    
    def epoch_verbose(self, epoch, **kwargs):
        log = "\nEpoch: {}/{} summary:".format(epoch, self.epochs)
        for k, v in kwargs.items():
            log += "\n            {}  |  {:1.6e}".format(k,v)
        print(log)
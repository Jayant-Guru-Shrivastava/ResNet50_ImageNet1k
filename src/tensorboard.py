# Install tensorboard if not already installed
import json
from torch.utils.tensorboard import SummaryWriter
import os

# Create a directory for logs
log_dir = './tensorboard_logs'
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Read and parse the JSONL file
with open('log.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        epoch = data['epoch']
        
        # Log training metrics
        if 'train_loss' in data:
            writer.add_scalar('Loss/train', data['train_loss'], epoch)
        if 'train_top1' in data:
            writer.add_scalar('Accuracy/train_top1', data['train_top1'], epoch)
        if 'train_top5' in data:
            writer.add_scalar('Accuracy/train_top5', data['train_top5'], epoch)
        
        # Log validation metrics
        if 'val_loss' in data:
            writer.add_scalar('Loss/val', data['val_loss'], epoch)
        if 'val_top1' in data:
            writer.add_scalar('Accuracy/val_top1', data['val_top1'], epoch)
        if 'val_top5' in data:
            writer.add_scalar('Accuracy/val_top5', data['val_top5'], epoch)
        
        # Log EMA metrics
        if 'ema_top1' in data:
            writer.add_scalar('EMA/top1', data['ema_top1'], epoch)
        if 'ema_top5' in data:
            writer.add_scalar('EMA/top5', data['ema_top5'], epoch)
        
        # Log learning rate
        if 'lr' in data:
            writer.add_scalar('Hyperparameters/learning_rate', data['lr'], epoch)
        
        # Log time per epoch
        if 'time_sec' in data:
            writer.add_scalar('Performance/time_per_epoch', data['time_sec'], epoch)

writer.close()

print("TensorBoard logs created successfully!")
print(f"Logs saved to: {log_dir}")
print("\nTo view in Colab, run:")
print("%load_ext tensorboard")
print(f"%tensorboard --logdir {log_dir}")

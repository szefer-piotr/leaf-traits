import torch

from typing import Callable, Dict, List
from tqdm.auto import tqdm

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    
    """Validates a PyTorch model for a single epoch.
    
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).
    
    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        
    Returns:
        A value of training loss.

    """

    model.train()
    
    train_loss = 0
    
    for batch, (X,y) in enumerate(dataloader):
        # print(f"Batch: {batch} of {len(dataloader)}")
        X['image']=X['image'].to(device)
        X['feature']=X['feature'].to(device) 
        y=y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred['targets'], y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = train_loss / len(dataloader)
    
    return train_loss



def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    
    """Validates a PyTorch model for a single epoch.
    
    Turns a target PyTorch model to 'eval' mode and then
    performs a forward pass on a testing dataset.
    
    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        
    Returns:
        A value of validation loss.

    """
    
    model.eval()
    
    val_loss = 0
    
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            # print(f"Batch: {batch}")
            X['image'], X['feature'], y= X['image'].to(device), X['feature'].to(device), y.to(device)
            val_preds = model(X)
            loss = loss_fn(val_preds['targets'], y)
            val_loss += loss.item()
    
    val_loss = val_loss / len(dataloader)
    
    return val_loss



def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epochs: int = 5,
    device: torch.device,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and validation_step()
    functions for a number of epochs, training and validating the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        trian_dataloader: A DataLoader instance for the model to be trained on.
        val_dataloader: A DataLoader instance for the model to be validated on.
        optimizer: A PyTorch oprimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" of "cpu").

    Returns:
        A dictionary of training and validation loss. Both metrics have values
        in a list for each epoch.
        In the form: {train_loss: [...], validation_loss: [...]}
    """
    
    results = {
        'train_loss': [],
        'validation_loss': []
    }

    for epoch in tqdm(range(epochs)):
        # print(epoch)
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        val_loss = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
        )

        results['train_loss'].append(train_loss)
        results['validation_loss'].append(val_loss)

    return results
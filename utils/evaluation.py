import torch

def get_test_accuracy(model, test_loader, model_name=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    
    print(f"\nEvaluating {model_name} model...")
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Handle different model outputs robustly (model may return logits or (logits, extra))
            result = model(batch_X)
            outputs = result[0] if isinstance(result, (tuple, list)) else result
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"{model_name} Test Accuracy: {accuracy:.2f}%")
    return accuracy
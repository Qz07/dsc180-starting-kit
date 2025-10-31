import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_all_outputs(net, loader):
    """
    Return (probs, labels, preds) for all samples in loader.
    Note: This function now *requires* the loader to return (inputs, labels).
    """
    net.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            logits = net(inputs)
            p = torch.softmax(logits, dim=1)
            
            all_probs.append(p.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_preds)

def compute_classification_truth_ratio(probs, labels):
    """
    Computes the ratio of P(true_label) / (P(true_label) + P(highest_wrong_label)).
    
    A value of 1.0 means high confidence in the true label.
    A value of 0.5 means the model is equally confident in the true label
    and the next-best-wrong label (i.e., it has forgotten).
    """
    # p_para: Get the probability of the *true label* for each sample
    # probs shape is (n_samples, n_classes), labels shape is (n_samples,)
    p_para = probs[np.arange(len(labels)), labels]
    
    # p_pert: Get the probability of the *highest-scoring wrong label*
    # We copy the probs, set the true label's prob to -1, and then find the max
    probs_copy = probs.copy()
    probs_copy[np.arange(len(labels)), labels] = -1.0
    p_pert = probs_copy.max(axis=1)
    
    # Compute the ratio, add epsilon for numerical stability
    ratio = p_para / (p_para + p_pert + 1e-12)
    return ratio.mean()

def truth_ratio(mdl, data_loader):
    forget_probs, forget_labels, _ = compute_all_outputs(mdl, data_loader)
    tr = compute_classification_truth_ratio(forget_probs, forget_labels)
    return tr
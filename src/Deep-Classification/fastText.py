import torch
from torchtext import data
from torchtext import datasets
import random
import data_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

EPOCH_SAVE = 10
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
BATCH_SIZE = 64
N_EPOCHS = 200000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
REPOSITORY_NAME = 'Yelp-reviews'

#SEED = 1234
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)

def train(model, iterator, optimizer, loss_function):
    
    epoch_loss = 0.
    epoch_acc = 0.
    epoch_real_prec = 0.
    epoch_fake_prec = 0.
    epoch_real_recall = 0.
    epoch_fake_recall = 0.
    epoch_real_f_score = 0.
    epoch_fake_f_score = 0.
    #epoch_roc = 0
    
    model.train()
    
    for batch_i, batch in enumerate(iterator):
        print ("Starting Batch: %d" % batch_i)

        optimizer.zero_grad()
        
        logits = model(batch.text).squeeze(1)
        
        loss = loss_function(logits, batch.label)
        
        acc = batch_accuracy(logits, batch.label)
        precisions, recalls, f1_scores = batch_precision_recall_f_score(logits, batch.label)
        #roc_score = batch_roc_score(logits, batch.label)

        print ("Batch %d accuracy: %f" % (batch_i, acc))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_real_prec += precisions[0]
        epoch_fake_prec += precisions[1]
        epoch_real_recall += recalls[0]
        epoch_fake_recall += recalls[1]
        epoch_real_f_score += f1_scores[0]
        epoch_fake_f_score += f1_scores[1]
        #epoch_roc += roc_score
    
    avg_prec = [epoch_real_prec / len(iterator), epoch_fake_prec / len(iterator)]
    avg_recall = [epoch_real_recall / len(iterator),  epoch_fake_recall / len(iterator)]
    avg_f_score = [epoch_real_f_score / len(iterator),  epoch_fake_f_score / len(iterator)]
    return epoch_loss / len(iterator), epoch_acc / len(iterator), avg_prec, avg_recall, avg_f_score #, epoch_roc / len(iterator)

def evaluate(model, iterator, loss_function):
    epoch_loss = 0.
    epoch_acc = 0.
    epoch_real_prec = 0.
    epoch_fake_prec = 0.
    epoch_real_recall = 0.
    epoch_fake_recall = 0.
    epoch_real_f_score = 0.
    epoch_fake_f_score = 0.
    #epoch_roc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = loss_function(predictions, batch.label)
            
            acc = batch_accuracy(predictions, batch.label)
            precisions, recalls, f1_scores = batch_precision_recall_f_score(predictions, batch.label)
            #roc_score = batch_roc_score(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_real_prec += precisions[0]
            epoch_fake_prec += precisions[1]
            epoch_real_recall += recalls[0]
            epoch_fake_recall += recalls[1]
            epoch_real_f_score += f1_scores[0]
            epoch_fake_f_score += f1_scores[1]
            #epoch_roc += roc_score
    
    avg_prec = [epoch_real_prec / len(iterator), epoch_fake_prec / len(iterator)]
    avg_recall = [epoch_real_recall / len(iterator),  epoch_fake_recall / len(iterator)]
    avg_f_score = [epoch_real_f_score / len(iterator),  epoch_fake_f_score / len(iterator)]
    return epoch_loss / len(iterator), epoch_acc / len(iterator), avg_prec, avg_recall, avg_f_score  #, epoch_roc / len(iterator)

def batch_accuracy(preds, y):
    # Pass through sigmoid with decision bound at 0.5
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

def batch_precision_recall_f_score(preds, y):
    y_pred = torch.round(torch.sigmoid(preds))
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred)
    return precisions, recalls, f1_scores

# def batch_roc_score(preds, y):
#     y_pred = torch.round(torch.sigmoid(preds))
#     y_pred = y_pred.detach().numpy()
#     y_true = y.detach().numpy() 
#     roc_score = roc_auc_score(y_true, y_pred)

def main():
    def generate_bigrams(x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

    # Load the data-set
    TEXT, train_itr, valid_itr, test_itr = data_loader.load_data(generate_bigrams)

    input_dim = len(TEXT.vocab)

    model = FastText(vocab_size=input_dim, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM)
    
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.BCEWithLogitsLoss()

    # Allow for running on GPU
    model = model.to(DEVICE)
    loss_function = loss_function.to(DEVICE)

    for epoch in range(1, N_EPOCHS):

        train_loss, train_acc, train_prec, train_recall, train_f_score = train(model, train_itr, optimizer, loss_function)
        valid_loss, valid_acc, valid_prec, valid_recall, valid_f_score = evaluate(model, valid_itr, loss_function)
        
        print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.2f}% | Train f1_score: {train_f_score*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Val. f1_score: {valid_f_score}%')
        saveMetrics('train', train_acc, train_loss, train_prec, train_recall, train_f_score, epoch)
        saveMetrics('valid', valid_acc, valid_loss, valid_prec, valid_recall, valid_f_score, epoch)
        if (epoch % EPOCH_SAVE == 0):
            saveModel(model, epoch)

    saveModel(model, epoch)

def setupCheckpoints():
    def get_repository_path():
        """ 
        Returns the path of the project repository

        Uses the global REPOSITORY_NAME constant and searches through parent directories
        """
        p = Path(__file__).absolute().parents
        for parent in p:
            if parent.name == REPOSITORY_NAME:
                return parent

    p = get_repository_path()
    checkpoints_folder = p / 'checkpoints'
    Fast_Text_folder = checkpoints_folder / 'FastText'
    cur_folder = Fast_Text_folder / CURRENT_TIME

    
    checkpoints_folder.mkdir(exist_ok=True)
    Fast_Text_folder.mkdir(exist_ok=True)
    cur_folder.mkdir(exist_ok=True)

    return cur_folder

def saveModel(model, epoch):
    path = setupCheckpoints()
    model_folder = path / 'models'
    model_folder.mkdir(exist_ok=True)

    model_path = model_folder / '{:02d}-model.pt'.format(epoch)

    state = model.state_dict()
    torch.save(state, model_path)

def saveMetrics(prefix, accuracy, loss, precisions, recalls, f1_scores, epoch):
    real_precision, fake_precision = precisions
    real_recall, fake_recall = recalls
    real_f1_score, fake_f1_score = f1_scores

    path = setupCheckpoints()

    accuracy_path = path / '{}-accuracy.txt'.format(prefix)
    loss_path = path / '{}-loss.txt'.format(prefix)

    real_precision_path = path / '{}-real-precision.txt'.format(prefix)
    fake_precision_path = path / '{}-fake-precision.txt'.format(prefix)
    real_recall_path = path / '{}-real-recall.txt'.format(prefix)
    fake_recall_path = path / '{}-fake-recall.txt'.format(prefix)
    real_f1_score_path = path / '{}-real-f1_score.txt'.format(prefix)
    fake_f1_score_path = path / '{}-fake-f1_score.txt'.format(prefix)

    def writeMetric(metric_path, value):
        with open(metric_path, 'a+') as f:
            f.write('{},{}\n'.format(epoch, value))

    writeMetric(accuracy_path, accuracy)
    writeMetric(loss_path, loss)
    writeMetric(real_precision_path, real_precision)
    writeMetric(fake_precision_path, fake_precision)
    writeMetric(real_recall_path, real_recall)
    writeMetric(fake_recall_path, fake_recall)
    writeMetric(real_f1_score_path, real_f1_score)
    writeMetric(fake_f1_score_path, fake_f1_score)

if __name__ == '__main__':
    main()

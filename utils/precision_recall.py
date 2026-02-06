import numpy as np

if __name__ == '__main__':
   from utils.data import load_preprocess
   from utils.model import load_model, predict

def precision_recall(y_probs, y_test, thresh: float = 0.5):
   
   y_preds = np.where(y_probs >= thresh, 1, 0)

   y_test = np.array(y_test).reshape(-1)
   
   ap_mask = (y_test == 1) # Obtaining indices of actual positives
   an_mask = (y_test == 0) # Obtaining indices of actual negatives
   pp_mask = (y_preds == 1) # Obtaining indices of predicted postives
   pn_mask = (y_preds == 0) # Obtaining indices of predicted negatives

   tp = np.sum((ap_mask * pp_mask))
   fp = np.sum((an_mask * pp_mask))
   tn = np.sum((an_mask * pn_mask))
   fn = np.sum((ap_mask * pn_mask))

   if __name__ == '__main__':
      output = f'\n\nTrue Positives: {tp}\nFalse Positives: {fp}\nTrue Negatives: {tn}\nFalse Negatives: {fn}\n\n'
      print(output)
         
   precision = 1 if np.isnan(tp / (tp + fp)) else tp / (tp + fp)

   recall = tp / (tp + fn)

   return precision, recall


def precision_recall_array(X, y, model, y_probs):
   
   pr_data = []
   
   for threshold in np.linspace(0,1,101):

      precision, recall = precision_recall(y_probs, y, threshold)

      pr_data.append([recall, precision])

   pr_data = np.array(pr_data) * 100

   return pr_data


if __name__ == '__main__':
   
   X, y = load_preprocess()
   model = load_model()
   y_probs = predict(model, X)

   precision, recall = precision_recall(y_probs, y, thresh = 1)

   print(f'Precision: {precision*100:.2f}%\nRecall: {recall*100:.2f}%')

   pr_data = precision_recall_array(X, y, model, y_probs)

   print(pr_data[:])

